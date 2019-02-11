import logging
import sys
import os

from automl.benchmark import TaskConfig
from automl.data import Dataset
from automl.datautils import Encoder, impute
from automl.results import save_predictions_to_file
from automl.utils import InterruptTimer, dir_of

os.environ['OMP_NUM_THREADS'] = '1'
sys.path.append("{}/libs/hyperopt-sklearn".format(dir_of(__file__)))
from hpsklearn import HyperoptEstimator, any_classifier, any_regressor
from hyperopt import tpe
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, log_loss, mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score

log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    log.info("\n**** Hyperopt-sklearn ****\n")

    is_classification = config.type == 'classification'

    default = lambda: 0
    metrics_to_loss_mapping = dict(
        acc=(default, False), # lambda y, pred: 1.0 - accuracy_score(y, pred)
        auc=(lambda y, pred: 1.0 - roc_auc_score(y, pred), False),
        f1=(lambda y, pred: 1.0 - f1_score(y, pred), False),
        # logloss=(log_loss, True),
        mae=(mean_absolute_error, False),
        mse=(mean_squared_error, False),
        msle=(mean_squared_log_error, False),
        r2=(default, False), # lambda y, pred: 1.0 - r2_score(y, pred)
    )
    loss_fn, continuous_loss_fn = metrics_to_loss_mapping[config.metric] if config.metric in metrics_to_loss_mapping else (None, False)
    if loss_fn is None:
        log.warning("Performance metric %s not supported: defaulting to %s.",
                    config.metric, 'accuracy' if is_classification else 'r2')
    if loss_fn is default:
        loss_fn = None

    log.warning("Ignoring cores constraint of %s cores.", config.cores)
    log.info("Running hyperopt-sklearn with a maximum time of %ss on %s cores, optimizing %s.",
             config.max_runtime_seconds, 'all', config.metric)

    X_train, X_test = impute(dataset.train.X_enc, dataset.test.X_enc)
    y_train, y_test = dataset.train.y_enc, dataset.test.y_enc

    if is_classification:
        classifier = any_classifier('clf')
        regressor = None
    else:
        classifier = None
        regressor = any_regressor('rgr')

    estimator = HyperoptEstimator(classifier=classifier,
                                  regressor=regressor,
                                  algo=tpe.suggest,
                                  loss_fn=loss_fn,
                                  continuous_loss_fn=continuous_loss_fn,
                                  trial_timeout=config.max_runtime_seconds,
                                  **config.framework_params)

    with InterruptTimer(config.max_runtime_seconds, kill_sub_processes=True):
        estimator.fit(X_train, y_train)

    predictions = estimator.predict(X_test)
    probabilities = Encoder('one-hot', target=False, encoded_type=float).fit_transform(predictions) if is_classification else None

    save_predictions_to_file(dataset=dataset,
                             output_file=config.output_predictions_file,
                             probabilities=probabilities,
                             predictions=predictions,
                             truth=y_test,
                             target_is_encoded=True)

