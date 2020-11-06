import logging
import os
import sys

sys.path.append("{}/lib/oboe/automl".format(os.path.realpath(os.path.dirname(__file__))))
from auto_learner import AutoLearner

from frameworks.shared.callee import call_run, result, utils

log = logging.getLogger(__name__)


def run(dataset, config):
    log.info("\n**** Oboe ****\n")

    is_classification = config.type == 'classification'
    if not is_classification:
        # regression currently fails (as of 26.02.2019: still under development state by oboe team)
        raise ValueError('Regression is not yet supported (under development).')

    X_train = dataset.train.X_enc
    y_train = dataset.train.y_enc

    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}
    n_cores = config.framework_params.get('_n_cores', config.cores)

    log.info('Running oboe with a maximum time of {}s on {} cores.'.format(config.max_runtime_seconds, n_cores))
    log.warning('We completely ignore the advice to optimize towards metric: {}.'.format(config.metric))

    aml = AutoLearner(p_type='classification' if is_classification else 'regression',
                      n_cores=n_cores,
                      runtime_limit=config.max_runtime_seconds,
                      **training_params)

    aml_models = lambda: [aml.ensemble, *aml.ensemble.base_learners] if len(aml.ensemble.base_learners) > 0 else []

    with utils.Timer() as training:
        try:
            aml.fit(X_train, y_train)
        except IndexError as e:
            if len(aml_models()) == 0:  # incorrect handling of some IndexError in oboe if ensemble is empty
                raise ValueError("Oboe could not produce any model in the requested time.")
            raise e

    log.info('Predicting on the test set.')
    X_test = dataset.test.X_enc
    y_test = dataset.test.y_enc
    predictions = aml.predict(X_test).reshape(len(X_test))

    if is_classification:
        probabilities = "predictions"  # encoding is handled by caller in `__init__.py`
    else:
        probabilities = None

    return result(output_file=config.output_predictions_file,
                  predictions=predictions,
                  truth=y_test,
                  probabilities=probabilities,
                  target_is_encoded=is_classification,
                  models_count=len(aml_models()),
                  training_duration=training.duration)


if __name__ == '__main__':
    call_run(run)
