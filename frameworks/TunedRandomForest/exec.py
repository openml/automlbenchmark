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

os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
import stopit

from frameworks.shared.callee import call_run, result, utils

log = logging.getLogger(__name__)


def pick_values_uniform(start: int, end: int, length: int):
    d = (end - start) / (length - 1)
    uniform_floats = [start + i * d for i in range(length)]
    return sorted(set([int(f) for f in uniform_floats]))


def run(dataset, config):
    log.info("\n**** Tuned Random Forest (sklearn) ****\n")

    is_classification = config.type == 'classification'

    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}
    tuning_params = config.framework_params.get('_tuning', training_params)
    n_jobs = config.framework_params.get('_n_jobs', config.cores)  # useful to disable multicore, regardless of the dataset config

    X_train, X_test = dataset.train.X_enc, dataset.test.X_enc
    y_train, y_test = dataset.train.y_enc, dataset.test.y_enc

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
    # Mix up the order of `max_features` to try, so that a fair range is tried even if we have too little time
    # to try all possible values. Order: [sqrt(p), 1, p, random order for remaining values]
    # max_features_to_try = below_default[1:] + above_default[:-1]
    # max_features_values = ([default_value, 1, n_features]
    #                        + random.sample(max_features_to_try, k=len(max_features_to_try)))
    max_features_values = [default_value] + below_default + above_default
    # Define up to how much of total time we spend 'optimizing' `max_features`.
    # (the remainder if used for fitting the final model).
    safety_factor = 0.85
    with stopit.ThreadingTimeout(seconds=int(config.max_runtime_seconds * safety_factor)):
        log.info("Evaluating multiple values for `max_features`: %s.", max_features_values)
        max_feature_scores = []
        tuning_durations = []
        for i, max_features_value in enumerate(max_features_values):
            log.info("[{:2d}/{:2d}] Evaluating max_features={}"
                     .format(i + 1, len(max_features_values), max_features_value))
            imputation = SimpleImputer()
            random_forest = estimator(n_jobs=n_jobs,
                                      random_state=config.seed,
                                      max_features=max_features_value,
                                      **tuning_params)
            pipeline = Pipeline(steps=[
                ('preprocessing', imputation),
                ('learning', random_forest)
            ])
            with utils.Timer() as cv_scoring:
                try:
                    scores = cross_val_score(estimator=pipeline,
                                             X=dataset.train.X_enc,
                                             y=dataset.train.y_enc,
                                             scoring=metric,
                                             cv=5)
                    max_feature_scores.append((statistics.mean(scores), max_features_value))
                except stopit.utils.TimeoutException as toe:
                    log.error("Failed CV scoring for max_features=%s : Timeout", max_features_value)
                    tuning_durations.append((max_features_value, cv_scoring.duration))
                    raise toe
                except Exception as e:
                    log.error("Failed CV scoring for max_features=%s :\n%s", max_features_value, e)
                    log.debug("Exception:", exc_info=True)
            tuning_durations.append((max_features_value, cv_scoring.duration))

    log.info("Tuning scores:\n%s", sorted(max_feature_scores))
    log.info("Tuning durations:\n%s", sorted(tuning_durations))
    _, best_max_features_value = max(max_feature_scores) if len(max_feature_scores) > 0 else (math.nan, 'auto')
    log.info("Training final model with `max_features={}`.".format(best_max_features_value))
    rf = estimator(n_jobs=n_jobs,
                   random_state=config.seed,
                   max_features=best_max_features_value,
                   **training_params)
    with utils.Timer() as training:
        rf.fit(X_train, y_train)

    predictions = rf.predict(X_test)
    probabilities = rf.predict_proba(X_test) if is_classification else None

    return result(
        output_file=config.output_predictions_file,
        predictions=predictions,
        truth=y_test,
        probabilities=probabilities,
        target_is_encoded=is_classification,
        models_count=len(rf),
        training_duration=training.duration+sum(map(lambda t: t[1], tuning_durations))
    )


if __name__ == '__main__':
    call_run(run)
