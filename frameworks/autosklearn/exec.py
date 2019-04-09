import logging
import os
import tempfile as tmp
import warnings

os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from autosklearn.estimators import AutoSklearnClassifier, AutoSklearnRegressor
import autosklearn.metrics as metrics

from automl.benchmark import TaskConfig
from automl.data import Dataset
from automl.results import save_predictions_to_file
from automl.utils import Timer, path_from_split, split_path

log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    log.info("\n**** AutoSklearn ****\n")
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=DeprecationWarning)

    is_classification = config.type == 'classification'

    # Mapping of benchmark metrics to autosklearn metrics
    metrics_mapping = dict(
        acc=metrics.accuracy,
        auc=metrics.roc_auc,
        f1=metrics.f1,
        logloss=metrics.log_loss,
        mae=metrics.mean_absolute_error,
        mse=metrics.mean_squared_error,
        r2=metrics.r2
    )
    perf_metric = metrics_mapping[config.metric] if config.metric in metrics_mapping else None
    if perf_metric is None:
        # TODO: figure out if we are going to blindly pass metrics through, or if we use a strict mapping
        log.warning("Performance metric %s not supported.", config.metric)

    # Set resources based on datasize
    log.info("Running auto-sklearn with a maximum time of %ss on %s cores with %sMB, optimizing %s.",
             config.max_runtime_seconds, config.cores, config.max_mem_size_mb, perf_metric)
    log.info("Environment: %s", os.environ)

    X_train = dataset.train.X_enc
    y_train = dataset.train.y_enc
    # log.info("finite=%s", np.isfinite(X_train))
    predictors_type = ['Categorical' if p.is_categorical() else 'Numerical' for p in dataset.predictors]

    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}

    cores = config.framework_params.get('_cores', config.cores)
    safety_memory_mb = config.framework_params.get('_safety_memory_mb', 0)  # 1024 (we already leave 2GB for OS)
    ensemble_memory_limit_mb = config.framework_params.get('_ensemble_memory_limit_mb', 1024)  # keeping defaults
    ml_memory_limit_mb = config.framework_params.get('_ml_memory_limit_mb', config.max_mem_size_mb) - safety_memory_mb
    log.info("Using %sM memory per ML job and %sM for ensemble job on a total of %s cores", ml_memory_limit_mb, ensemble_memory_limit_mb, cores)

    log.warning("Using meta-learned initialization, which might be bad (leakage).")
    # TODO: do we need to set per_run_time_limit too?
    estimator = AutoSklearnClassifier if is_classification else AutoSklearnRegressor
    auto_sklearn = estimator(time_left_for_this_task=config.max_runtime_seconds,
                             n_jobs=cores,
                             ml_memory_limit=ml_memory_limit_mb,
                             ensemble_memory_limit=ensemble_memory_limit_mb,
                             seed=config.seed,
                             **training_params)
    with Timer() as training:
        auto_sklearn.fit(X_train, y_train, metric=perf_metric, feat_type=predictors_type)

    models_repr = auto_sklearn.show_models()
    log.debug("Trained Ensemble:\n%s", models_repr)
    models_file = split_path(config.output_predictions_file)
    models_file.extension = '.models.txt'
    models_file = path_from_split(models_file)
    with open(models_file, 'w') as f:
        f.write(models_repr)

    # Convert output to strings for classification
    log.info("Predicting on the test set.")
    X_test= dataset.test.X_enc
    y_test = dataset.test.y_enc
    predictions = auto_sklearn.predict(X_test)
    probabilities = auto_sklearn.predict_proba(X_test) if is_classification else None

    save_predictions_to_file(dataset=dataset,
                             output_file=config.output_predictions_file,
                             probabilities=probabilities,
                             predictions=predictions,
                             truth=y_test,
                             target_is_encoded=True)

    return dict(
        models_count=len(auto_sklearn.get_models_with_weights()),
        training_duration=training.duration
    )
