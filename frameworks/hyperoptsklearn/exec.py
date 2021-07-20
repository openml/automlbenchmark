import functools as ft
import logging
import math
import os
import signal
import tempfile as tmp

os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from hpsklearn import HyperoptEstimator, any_classifier, any_regressor
import hyperopt
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, log_loss, mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score

from frameworks.shared.callee import call_run, result
from frameworks.shared.utils import InterruptTimeout, Timer, dir_of, kill_proc_tree

log = logging.getLogger(__name__)


def run(dataset, config):
    log.info(f"\n**** Hyperopt-sklearn [v{config.framework_version}] ****\n")

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
        rmse=(mean_squared_error, False),
    )
    loss_fn, continuous_loss_fn = metrics_to_loss_mapping[config.metric] if config.metric in metrics_to_loss_mapping else (None, False)
    if loss_fn is None:
        log.warning("Performance metric %s not supported: defaulting to %s.",
                    config.metric, 'accuracy' if is_classification else 'r2')
    if loss_fn is default:
        loss_fn = None

    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}
    if 'algo' in training_params:
        training_params['algo'] = eval(training_params['algo'])  # evil eval: use get_extensions instead once https://github.com/openml/automlbenchmark/pull/141 is merged

    log.warning("Ignoring cores constraint of %s cores.", config.cores)
    log.info("Running hyperopt-sklearn with a maximum time of %ss on %s cores, optimizing %s.",
             config.max_runtime_seconds, 'all', config.metric)

    X_train = dataset.train.X
    y_train = dataset.train.y

    if is_classification:
        classifier = any_classifier('clf')
        regressor = None
    else:
        classifier = None
        regressor = any_regressor('rgr')

    estimator = HyperoptEstimator(classifier=classifier,
                                  regressor=regressor,
                                  loss_fn=loss_fn,
                                  continuous_loss_fn=continuous_loss_fn,
                                  trial_timeout=config.max_runtime_seconds,
                                  seed=config.seed,
                                  **training_params)

    with InterruptTimeout(config.max_runtime_seconds,
                          interruptions=[
                              dict(),  # default interruption
                              dict(sig=signal.SIGKILL)
                          ],
                          wait_retry_secs=math.ceil(config.max_runtime_seconds/60),
                          before_interrupt=ft.partial(kill_proc_tree, timeout=5, include_parent=False)):
        with Timer() as training:
            estimator.fit(X_train, y_train)

    log.info('Predicting on the test set.')
    X_test = dataset.test.X
    y_test = dataset.test.y
    with Timer() as predict:
        predictions = estimator.predict(X_test)

    if is_classification:
        probabilities = "predictions"  # encoding is handled by caller in `__init__.py`
    else:
        probabilities = None

    return result(output_file=config.output_predictions_file,
                  predictions=predictions,
                  truth=y_test,
                  probabilities=probabilities,
                  target_is_encoded=is_classification,
                  models_count=len(estimator.trials),
                  training_duration=training.duration,
                  predict_duration=predict.duration)


if __name__ == '__main__':
    call_run(run)
