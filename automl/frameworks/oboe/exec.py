import logging
import sys
sys.path.append("./libs/oboe/automl")

from auto_learner import AutoLearner

from automl.benchmark import TaskConfig
from automl.data import Dataset
from automl.datautils import Encoder, impute
from automl.results import save_predictions_to_file


log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    log.info("\n**** Oboe ****\n")

    X_train, X_test = impute(dataset.train.X_enc, dataset.test.X_enc)
    y_train, y_test = dataset.train.y_enc, dataset.test.y_enc

    log.info('Running oboe with a maximum time of {}s on {} cores.'.format(config.max_runtime_seconds, config.cores))
    log.warning('We completely ignore the advice to optimize towards metric: {}.'.format(config.metric))

    automl = AutoLearner(p_type='classification',
                         n_cores=config.cores,
                         runtime_limit=config.max_runtime_seconds)
    automl.fit(X_train, y_train)
    class_predictions = automl.predict(X_test).reshape(len(X_test))
    class_probabilities = Encoder('one-hot', target=False).fit_transform(class_predictions)

    save_predictions_to_file(dataset=dataset,
                             output_file=config.output_predictions_file,
                             class_probabilities=class_probabilities,
                             class_predictions=class_predictions,
                             class_truth=y_test,
                             classes_are_encoded=True)


