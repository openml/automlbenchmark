import logging
import time

from tpot import TPOTClassifier, TPOTRegressor

from automl.benchmark import TaskConfig
from automl.data import Dataset
from automl.datautils import Encoder, impute
from automl.results import save_predictions_to_file


log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    log.info("\n**** TPOT ****\n")

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
        r2='r2'
    )
    scoring_metric = metrics_mapping[config.metric] if config.metric in metrics_mapping else None
    if scoring_metric is None:
        raise ValueError("Performance metric {} not supported.".format(config.metric))

    X_train, X_test = impute(dataset.train.X_enc, dataset.test.X_enc)
    y_train, y_test = dataset.train.y_enc, dataset.test.y_enc

    log.info('Running TPOT with a maximum time of %ss on %s cores, optimizing %s.',
             config.max_runtime_seconds, config.cores, scoring_metric)
    runtime_min = (config.max_runtime_seconds/60)

    estimator = TPOTClassifier if is_classification else TPOTRegressor
    tpot = estimator(n_jobs=config.cores,
                     max_time_mins=runtime_min,
                     verbosity=2,
                     scoring=scoring_metric,
                     **config.framework_params)

    start_time = time.time()
    tpot.fit(X_train, y_train)
    actual_runtime_min = (time.time() - start_time)/60.0
    log.debug('Requested training time: %sm.', runtime_min)
    log.debug('Actual training time: %sm.', actual_runtime_min)

    log.info('Predicting on the test set.')
    class_predictions = tpot.predict(X_test)
    try:
        class_probabilities = tpot.predict_proba(X_test) if is_classification else None
    except RuntimeError:
        # TPOT throws a RuntimeError if the optimized pipeline does not support `predict_proba`.
        class_probabilities = Encoder('one-hot', target=False).fit_transform(class_predictions)

    save_predictions_to_file(dataset=dataset,
                             output_file=config.output_predictions_file,
                             class_probabilities=class_probabilities,
                             class_predictions=class_predictions,
                             class_truth=y_test,
                             classes_are_encoded=is_classification)

