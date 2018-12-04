import logging

from auto_learner import AutoLearner

from automl.benchmark import TaskConfig
from automl.data import Dataset
from automl.datautils import Encoder
from automl.results import save_predictions_to_file


log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    log.info("\n**** Oboe ****\n")

    X_train = dataset.train.X_enc.astype(float)
    y_train = dataset.train.y_enc
    X_test = dataset.test.X_enc.astype(float)
    y_test = dataset.test.y_enc

    log.info('Running oboe with a maximum time of {}s on {} cores.'.format(config.max_runtime_seconds, config.cores))
    log.warning('We completely ignore the advice to optimize towards metric: {}.'.format(config.metric))

    automl = AutoLearner(p_type='classification',
                         n_cores=config.cores,
                         runtime_limit=config.max_runtime_seconds
                         )
    automl.fit_doubling_time_constrained(X_train, y_train)
    class_predictions = automl.predict(X_test)
    class_probabilities = Encoder('one-hot').fit_transform(class_predictions).astype(float)

    save_predictions_to_file(dataset=dataset,
                             output_file=config.output_predictions_file,
                             class_probabilities=class_probabilities,
                             class_predictions=class_predictions,
                             class_truth=y_test,
                             classes_are_encoded=True)


