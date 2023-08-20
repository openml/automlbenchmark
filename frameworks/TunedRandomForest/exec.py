"""
This 'system' first determines the best value for `max_features` for the Random Forest,
by trying up to 10 (uniformly distributed) values of 1..sqrt(p)...p. (p = number of features of the data).
It produces predictions based on a model trained with all of the data for the best found `max_features` value.
"""
import logging
import math
import os
import statistics
import tempfile as tmp
from collections import defaultdict
from typing import List

os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import psutil
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from frameworks.shared.callee import call_run, result, measure_inference_times
from frameworks.shared.utils import Timer
from custom_validate import cross_validate

log = logging.getLogger(__name__)


def pick_values_uniform(start: int, end: int, length: int):
    d = (end - start) / (length - 1)
    uniform_floats = [start + i * d for i in range(length)]
    return sorted(set([int(f) for f in uniform_floats]))


def extrapolate_with_worst_case(values: List[float], n: int = 5) -> float:
    """ Extrapolate the next value for `values`, based on the last `n` samples. """
    n = min(len(values), n)
    return values[-1] + max(v_next - v_prev for v_prev, v_next in zip(values[-n:], values[-n+1:]))


def run(dataset, config):
    log.info(f"\n**** Tuned Random Forest [sklearn v{sklearn.__version__}] ****\n")

    is_classification = config.type == 'classification'

    training_params = {
        k: v for k, v in config.framework_params.items()
        if not (k.startswith('_') or k == "n_estimators")
    }
    if "max_features" in training_params:
        raise ValueError("`max_features` may not be specified for Tuned Random Forest.")

    n_jobs = config.framework_params.get('_n_jobs', config.cores)  # useful to disable multicore, regardless of the dataset config
    k_folds = config.framework_params.get('_k_folds', 5)
    step_size = config.framework_params.get('_step_size', 10)
    memory_margin = config.framework_params.get('_memory_margin', 0.9)
    final_forest_size = config.framework_params.get('n_estimators', 2000)
    tune_fraction = config.framework_params.get('_tune_fraction', 0.9)
    tune_time = config.max_runtime_seconds * tune_fraction

    X_train, X_test = dataset.train.X, dataset.test.X
    y_train, y_test = dataset.train.y, dataset.test.y

    log.info("Running RandomForest with a maximum time of {}s on {} cores."
             .format(config.max_runtime_seconds, n_jobs))

    estimator = RandomForestClassifier if is_classification else RandomForestRegressor
    metric = dict(
        acc='accuracy',
        auc='roc_auc',
        f1='f1',
        logloss='neg_log_loss',
        mae='neg_mean_absolute_error',
        mse='neg_mean_squared_error',
        r2='r2',
        rmse='neg_root_mean_squared_error',
    )[config.metric]

    n_features = X_train.shape[1]
    default_value = max(1, int(math.sqrt(n_features)))
    below_default = pick_values_uniform(start=1, end=default_value, length=5+1)[:-1]   # 5 below
    above_default = pick_values_uniform(start=default_value, end=n_features, length=10+1 - len(below_default))[1:]  # 5 above

    max_features_values = [default_value] + below_default + above_default
    # Define up to how much of total time we spend 'optimizing' `max_features`.
    # (the remainder if used for fitting the final model).

    log.info("Evaluating multiple values for `max_features`: %s.", max_features_values)
    max_feature_scores = defaultdict(list)
    tuning_durations = defaultdict(list)
    memory_usage_by = defaultdict(list)
    this_process = psutil.Process(os.getpid())
    last_initial_fit_time = 0

    with Timer() as training:
        while max_features_values:
            time_left = tune_time - training.duration
            time_per_value = time_left / len(max_features_values)
            if time_per_value < last_initial_fit_time:
                log.info("Expect to exceed time constraints on next first fit, "
                         f"budget is {time_per_value}s and last first fit took "
                         f"{last_initial_fit_time}s.")
                log.info(f"Did not try max_features={max_features_values}.")
                break

            value = max_features_values.pop(0)
            log.info(f"Evaluating max_features={value} in {time_per_value} seconds.")

            random_forests = [
                estimator(
                    n_jobs=n_jobs,
                    random_state=config.seed,
                    max_features=value,
                    n_estimators=step_size,
                    warm_start=True,
                    **training_params
                ) for _ in range(k_folds)
            ]

            training_times = [training.duration]
            memory_usage = [this_process.memory_info()[0] / (2 ** 20)]

            while True:
                try:
                    cv_result = cross_validate(
                        estimators=random_forests,
                        X=X_train,
                        y=y_train,
                        scoring=metric,
                        error_score='raise',
                        cv=k_folds,
                        return_estimator=True,
                    )
                    max_feature_scores[value].append(statistics.mean(cv_result["test_score"]))
                except Exception as e:
                    log.error("Failed CV scoring for max_features=%s :\n%s", value, e)
                    log.debug("Exception:", exc_info=True)
                    max_feature_scores[value].append(math.nan)
                    break
                finally:
                    training_times.append(training.duration)
                    memory_usage.append(this_process.memory_info()[0] / (2 ** 20))

                if random_forests[0].n_estimators >= final_forest_size:
                    log.info("Stop training because desired forest size has been reached.")
                    break
                if extrapolate_with_worst_case(training_times) - training_times[0] >= time_per_value:
                    log.info(f"Stop training after fitting {random_forests[0].n_estimators} trees because it expects to exceed its time budget.")
                    break
                elif extrapolate_with_worst_case(memory_usage) >= config.max_mem_size_mb * memory_margin:
                    log.info(f"Stop training after fitting {random_forests[0].n_estimators} trees because it expects to exceed its memory budget.")
                    break

                for rf in random_forests:
                    rf.n_estimators += step_size

            tuning_durations[value] = training_times
            memory_usage_by[value] = memory_usage
            last_initial_fit_time = training_times[1] - training_times[0]

        summary = []
        for max_feature, scores in max_feature_scores.items():
            # remember: duration and memory have a record before the first fit too
            stats_for_max_feature = zip(scores, tuning_durations[max_feature][1:], memory_usage_by[max_feature][1:])
            for i, (score, duration, memory) in enumerate(stats_for_max_feature, start=1):
                summary.append({
                    "max_features": max_feature,
                    "trees": (step_size) * i,
                    metric: score,
                    "duration total (s)": duration,
                    "memory total (mb)": memory,
                    "duration delta (s)": duration - tuning_durations[max_feature][i - 1],
                    "memory delta (mb)": memory - memory_usage_by[max_feature][i - 1],
                })
        summary = pd.DataFrame.from_records(summary)
        log.debug("Report of tuning iterations:\n %s \n" % summary.to_string())
        log.info("Report final tunings:\n %s \n" % summary.groupby("max_features").last().sort_values(by="duration total (s)").to_string())

        # Iteratively fit a random forest with our "optimal" `max_features`
        _, best_value = max((scores[-1], value) for value, scores in max_feature_scores.items())
        log.info("Training final model with `max_features=%s`.", best_value)
        rf = estimator(n_jobs=n_jobs,
                       random_state=config.seed,
                       max_features=best_value,
                       warm_start=True,
                       n_estimators=step_size,
                       **training_params)

        training_times = [training.duration]
        memory_usage = [this_process.memory_info()[0] / (2 ** 20)]
        while True:
            rf.fit(X_train, y_train)
            training_times.append(training.duration)
            memory_usage.append(this_process.memory_info()[0] / (2 ** 20))
            if rf.n_estimators == final_forest_size:
                log.info("Stop training because desired forest size has been reached.")
                break
            if extrapolate_with_worst_case(training_times) >= config.max_runtime_seconds:
                log.info("Stop training because it expects to exceed its time budget.")
                break
            elif extrapolate_with_worst_case(
                    memory_usage) >= config.max_mem_size_mb * memory_margin:
                log.info(
                    "Stop training because it expects to exceed its memory budget.")
                break
            rf.n_estimators += step_size
    log.info(f"Finished fit in {training.duration}s.")

    with Timer() as predict:
        predictions = rf.predict(X_test)
    probabilities = rf.predict_proba(X_test) if is_classification else None
    log.info(f"Finished predict in {predict.duration}s.")

    def infer(data):
        data = pd.read_parquet(data) if isinstance(data, str) else data
        return rf.predict(data)

    inference_times = {}
    if config.measure_inference_time:
        inference_times["file"] = measure_inference_times(infer, dataset.inference_subsample_files)
        test_data = X_test if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test)
        inference_times["df"] = measure_inference_times(
            infer,
            [(1, test_data.sample(1, random_state=i)) for i in range(100)],
        )
        log.info(f"Finished inference time measurements.")

    return result(
        output_file=config.output_predictions_file,
        predictions=predictions,
        truth=y_test,
        probabilities=probabilities,
        target_is_encoded=is_classification,
        models_count=len(rf),
        training_duration=training.duration,
        predict_duration=predict.duration,
        inference_times=inference_times,
    )


if __name__ == '__main__':
    call_run(run)
