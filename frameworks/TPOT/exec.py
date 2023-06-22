import logging
import os
import pprint
import sys
import tempfile as tmp

import pandas as pd
from numpy.random import default_rng

if sys.platform == 'darwin':
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from tpot import TPOTClassifier, TPOTRegressor, __version__

from frameworks.shared.callee import call_run, output_subdir, result, \
    measure_inference_times
from frameworks.shared.utils import Timer, is_sparse


log = logging.getLogger(__name__)


def run(dataset, config):
    log.info(f"\n**** TPOT [v{__version__}]****\n")

    is_classification = config.type == 'classification'
    # Mapping of benchmark metrics to TPOT metrics
    metrics_mapping = dict(
        acc='accuracy',
        auc='roc_auc',
        f1='f1',
        logloss='neg_log_loss',
        mae='neg_mean_absolute_error',
        mse='neg_mean_squared_error',
        msle='neg_mean_squared_log_error',
        r2='r2',
        rmse='neg_mean_squared_error',  # TPOT can score on mse, as app computes rmse independently on predictions
    )
    scoring_metric = metrics_mapping[config.metric] if config.metric in metrics_mapping else None
    if scoring_metric is None:
        raise ValueError("Performance metric {} not supported.".format(config.metric))

    X_train = dataset.train.X
    y_train = dataset.train.y

    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}
    n_jobs = config.framework_params.get('_n_jobs', config.cores)  # useful to disable multicore, regardless of the dataset config
    config_dict = config.framework_params.get('_config_dict', "TPOT sparse" if is_sparse(X_train) else None)

    log.info('Running TPOT with a maximum time of %ss on %s cores, optimizing %s.',
             config.max_runtime_seconds, n_jobs, scoring_metric)
    runtime_min = (config.max_runtime_seconds/60)

    estimator = TPOTClassifier if is_classification else TPOTRegressor
    tpot = estimator(n_jobs=n_jobs,
                     max_time_mins=runtime_min,
                     scoring=scoring_metric,
                     random_state=config.seed,
                     config_dict=config_dict,
                     **training_params)

    with Timer() as training:
        tpot.fit(X_train, y_train)
    log.info(f"Finished fit in {training.duration}s.")


    def infer(data):
        data = pd.read_parquet(data) if isinstance(data, str) else data
        if is_classification:
            try:
                return tpot.predict_proba(data)
            except (RuntimeError, AttributeError):
                return tpot.predict(data)
        return tpot.predict(data)

    inference_times = {}
    if config.measure_inference_time:
        log.info("TPOT inference time measurements exclude preprocessing time of AMLB.")
        inference_times["file"] = measure_inference_times(infer, dataset.inference_subsample_files)
        inference_times["df"] = measure_inference_times(
            infer, [
                (1, dataset.test.X[default_rng(seed=i).integers(len(dataset.test.X)), :].reshape(1, -1))
                for i in range(100)
            ],
        )
        log.info(f"Finished inference time measurements.")

    log.info('Predicting on the test set.')
    y_test = dataset.test.y
    with Timer() as predict:
        X_test = dataset.test.X
        predictions = tpot.predict(X_test)

    try:
        probabilities = tpot.predict_proba(X_test) if is_classification else None
    except (RuntimeError, AttributeError):
        # TPOT throws a RuntimeError or AttributeError if the optimized pipeline
        # does not support `predict_proba` (which one depends on the version).
        probabilities = "predictions"  # encoding is handled by caller in `__init__.py`

    log.info(f"Finished predict in {predict.duration}s.")
    save_artifacts(tpot, config)

    return result(output_file=config.output_predictions_file,
                  predictions=predictions,
                  truth=y_test,
                  probabilities=probabilities,
                  target_is_encoded=is_classification,
                  models_count=len(tpot.evaluated_individuals_),
                  training_duration=training.duration,
                  predict_duration=predict.duration,
                  inference_times=inference_times,
                  )


def save_artifacts(estimator, config):
    try:
        log.debug("All individuals :\n%s", list(estimator.evaluated_individuals_.items()))
        models = estimator.pareto_front_fitted_pipelines_
        hall_of_fame = list(zip(reversed(estimator._pareto_front.keys), estimator._pareto_front.items))
        artifacts = config.framework_params.get('_save_artifacts', False)
        if 'models' in artifacts:
            models_file = os.path.join(output_subdir('models', config), 'models.txt')
            with open(models_file, 'w') as f:
                for m in hall_of_fame:
                    pprint.pprint(dict(
                        fitness=str(m[0]),
                        model=str(m[1]),
                        pipeline=models[str(m[1])],
                    ), stream=f)
    except Exception:
        log.debug("Error when saving artifacts.", exc_info=True)


if __name__ == '__main__':
    call_run(run)
