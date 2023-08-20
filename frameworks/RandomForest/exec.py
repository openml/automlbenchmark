import logging
import os
import tempfile as tmp
from typing import List

import pandas as pd

os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import psutil
import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from frameworks.shared.callee import call_run, result, measure_inference_times
from frameworks.shared.utils import Timer

log = logging.getLogger(os.path.basename(__file__))


def extrapolate_with_worst_case(values: List[float], n: int = 5) -> float:
    """ Extrapolate the next value for `values`, based on the last `n` samples. """
    n = min(len(values), n)
    return values[-1] + max(v_next - v_prev for v_prev, v_next in zip(values[-n:], values[-n + 1:]))


def run(dataset, config):
    log.info(f"\n**** Random Forest [sklearn v{sklearn.__version__}] ****\n")

    is_classification = config.type == 'classification'
    this_process = psutil.Process(os.getpid())

    encode = config.framework_params.get('_encode', True)
    X_train, X_test = dataset.train.X, dataset.test.X
    y_train, y_test = dataset.train.y, dataset.test.y

    training_params = {
        k: v for k, v in config.framework_params.items()
        if not (k.startswith('_') or k == "n_estimators")
    }
    n_jobs = config.framework_params.get('_n_jobs', config.cores)  # useful to disable multicore, regardless of the dataset config
    step_size = config.framework_params.get('_step_size', 10)
    final_forest_size = config.framework_params.get('n_estimators', 2000)

    # Default margins are conservative, because robustness is paramount for a baseline.
    time_margin = config.framework_params.get('_time_margin', 0.9)
    memory_margin = config.framework_params.get('_memory_margin', 0.9)

    log.info("Running RandomForest with a maximum time of {}s on {} cores.".format(config.max_runtime_seconds, n_jobs))
    log.warning("We completely ignore the advice to optimize towards metric: {}.".format(config.metric))

    estimator = RandomForestClassifier if is_classification else RandomForestRegressor
    rf = estimator(n_jobs=n_jobs,
                   random_state=config.seed,
                   n_estimators=step_size,
                   warm_start=True,
                   **training_params)

    with Timer() as training:
        training_times = [training.duration]
        memory_usage = [this_process.memory_info()[0] / (2**20)]

        while True:
            rf.fit(X_train, y_train)

            training_times.append(training.duration)
            memory_usage.append(this_process.memory_info()[0] / (2**20))
            log.info(f"Model trained {len(rf.estimators_):6d} trees in {int(training_times[-1]):6d} seconds using {int(memory_usage[-1]):6d}mb memory.")

            will_run_out_of_memory = extrapolate_with_worst_case(memory_usage) >= config.max_mem_size_mb * memory_margin
            will_run_out_of_time = extrapolate_with_worst_case(training_times) >= config.max_runtime_seconds * time_margin
            if rf.n_estimators >= final_forest_size:
                log.info("Stop training because desired forest size has been reached.")
                break
            elif will_run_out_of_time:
                log.info("Stop training because it expects to exceed its time budget.")
                break
            elif will_run_out_of_memory:
                log.info("Stop training because it expects to exceed its memory budget.")
                break
            else:
                # https://stackoverflow.com/questions/42757892/how-to-use-warm-start/42763502
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


    return result(output_file=config.output_predictions_file,
                  predictions=predictions,
                  truth=y_test,
                  probabilities=probabilities,
                  target_is_encoded=encode,
                  models_count=len(rf),
                  training_duration=training.duration,
                  predict_duration=predict.duration,
                  inference_times=inference_times,)


if __name__ == '__main__':
    call_run(run)
