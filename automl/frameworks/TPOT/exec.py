import logging
import time

from tpot import TPOTClassifier

from automl.benchmark import TaskConfig
from automl.data import Dataset
from automl.datautils import Encoder
from automl.results import save_predictions_to_file


log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    log.info("\n**** TPOT ****\n")

    # Mapping of benchmark metrics to TPOT metrics
    if config.metric == 'acc':
        metric = 'accuracy'
    elif config.metric == 'auc':
        metric = 'roc_auc'
    elif config.metric == 'logloss':
        metric = 'neg_log_loss'
    else:
        raise ValueError("Performance metric {} not supported.".format(config.metric))

    X_train = dataset.train.X_enc.astype(float)
    y_train = dataset.train.y_enc
    X_test = dataset.test.X_enc.astype(float)
    y_test = dataset.test.y_enc

    log.info('Running TPOT with a maximum time of {}s on {} cores, optimizing {}.'
          .format(config.max_runtime_seconds, config.cores, metric))

    runtime_min = (config.max_runtime_seconds/60)
    tpot = TPOTClassifier(n_jobs=config.cores,
                          max_time_mins=runtime_min,
                          verbosity=2,
                          scoring=metric)
    start_time = time.time()
    tpot.fit(X_train, y_train)
    actual_runtime_min = (time.time() - start_time)/60.0
    log.debug('Requested training time (minutes): ' + str(runtime_min))
    log.debug('Actual training time (minutes): ' + str(actual_runtime_min))

    log.info('Predicting on the test set.')
    class_predictions = tpot.predict(X_test)
    try:
        class_probabilities = tpot.predict_proba(X_test)
    except RuntimeError:
        # TPOT throws a RuntimeError if the optimized pipeline does not support `predict_proba`.
        class_probabilities = Encoder('one-hot').fit_transform(class_predictions).astype(float)

    save_predictions_to_file(dataset=dataset,
                             output_file=config.output_predictions_file,
                             class_probabilities=class_probabilities,
                             class_predictions=class_predictions,
                             class_truth=y_test,
                             classes_are_encoded=True)

