import logging
import sys

from automl.benchmark import TaskConfig
from automl.data import Dataset
from automl.datautils import Encoder, impute
from automl.results import save_predictions_to_file
from automl.utils import dir_of

sys.path.append("{}/libs/oboe/automl".format(dir_of(__file__)))
from auto_learner import AutoLearner

log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    log.info("\n**** Oboe ****\n")

    is_classification = config.type == 'classification'
    if not is_classification:
        # regression currently fails (as of 29.01.2019: still under development state by oboe team)
        raise ValueError('Regression is not yet supported (under development).')

    X_train, X_test = impute(dataset.train.X_enc, dataset.test.X_enc)
    y_train, y_test = dataset.train.y_enc, dataset.test.y_enc

    log.info('Running oboe with a maximum time of {}s on {} cores.'.format(config.max_runtime_seconds, config.cores))
    log.warning('We completely ignore the advice to optimize towards metric: {}.'.format(config.metric))

    automl = AutoLearner(p_type='classification' if is_classification else 'regression',
                         n_cores=config.cores,
                         runtime_limit=config.max_runtime_seconds,
                         **config.framework_params)
    automl.fit(X_train, y_train)
    predictions = automl.predict(X_test).reshape(len(X_test))
    probabilities = Encoder('one-hot', target=False, encoded_type=float).fit_transform(predictions) if is_classification else None

    save_predictions_to_file(dataset=dataset,
                             output_file=config.output_predictions_file,
                             probabilities=probabilities,
                             predictions=predictions,
                             truth=y_test,
                             target_is_encoded=True)


