import logging
import os
import sys
import tempfile as tmp

if sys.platform == 'darwin':
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'


import category_encoders
from packaging import version
import sklearn

from gama.data_loading import file_to_pandas
from gama import GamaClassifier, GamaRegressor, __version__

from frameworks.shared.callee import call_run, result, output_subdir
from frameworks.shared.utils import Timer, touch


log = logging.getLogger(__name__)


def run(dataset, config):
    log.info("\n**** GAMA [v%s] ****", __version__)
    log.info("sklearn == %s", sklearn.__version__)
    log.info("category_encoders == %s", category_encoders.__version__)

    is_classification = (config.type == 'classification')
    # Mapping of benchmark metrics to GAMA metrics
    metrics_mapping = dict(
        acc='accuracy',
        auc='roc_auc',
        f1='f1',
        logloss='neg_log_loss',
        mae='neg_mean_absolute_error',
        mse='neg_mean_squared_error',
        msle='neg_mean_squared_log_error',
        r2='r2',
        rmse='neg_mean_squared_error',
    )
    scoring_metric = metrics_mapping[config.metric] if config.metric in metrics_mapping else None
    if scoring_metric is None:
        raise ValueError("Performance metric {} not supported.".format(config.metric))

    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}
    n_jobs = config.framework_params.get('_n_jobs', config.cores)  # useful to disable multicore, regardless of the dataset config

    log.info('Running GAMA with a maximum time of %ss on %s cores, optimizing %s.',
             config.max_runtime_seconds, n_jobs, scoring_metric)

    estimator = GamaClassifier if is_classification else GamaRegressor
    kwargs = dict(
        n_jobs=n_jobs,
        max_total_time=config.max_runtime_seconds,
        scoring=scoring_metric,
        random_state=config.seed,
        **training_params
    )
    version_leq_20_2_0 = version.parse(__version__) <= version.parse('20.2.0')
    if version_leq_20_2_0:
        log_file = touch(os.path.join(output_subdir('logs', config), 'gama.log'))
        kwargs['keep_analysis_log'] = log_file
    else:
        kwargs['max_memory_mb'] = config.max_mem_size_mb
        kwargs['output_directory'] = output_subdir('logs', config)
    
    gama_automl = estimator(**kwargs)

    X_train, y_train = dataset.train.X, dataset.train.y
    # data = file_to_pandas(dataset.train.path, encoding='utf-8')
    # X_train, y_train = data.loc[:, data.columns != dataset.target], data.loc[:, dataset.target]

    with Timer() as training_timer:
        gama_automl.fit(X_train, y_train)

    log.info('Predicting on the test set.')
    X_test, y_test = dataset.test.X, dataset.test.y
    # data = file_to_pandas(dataset.test.path, encoding='utf-8')
    # X_test, y_test = data.loc[:, data.columns != dataset.target], data.loc[:, dataset.target]

    with Timer() as predict_timer:
        predictions = gama_automl.predict(X_test)
    if is_classification:
        probabilities = gama_automl.predict_proba(X_test)
    else:
        probabilities = None

    return result(
        output_file=config.output_predictions_file,
        predictions=predictions,
        probabilities=probabilities,
        truth=y_test,
        target_is_encoded=False,
        models_count=len(gama_automl._final_pop),
        training_duration=training_timer.duration,
        predict_duration=predict_timer.duration
    )


if __name__ == '__main__':
    call_run(run)
