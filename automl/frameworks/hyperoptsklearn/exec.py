import logging
import os

from hpsklearn import HyperoptEstimator, any_classifier
from hyperopt import tpe

from automl.benchmark import TaskConfig
from automl.data import Dataset
from automl.utils import one_hot_encode_predictions, save_predictions_to_file


log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    log.info("\n**** Hyperopt-sklearn ****\n")

    log.info("Running hyperopt-sklearn with a maximum time of {}s on {} cores, optimizing {}"
            .format(config.max_runtime_seconds, config.cores, config.metric))

    X_train = dataset.train.X_enc.astype(float)
    y_train = dataset.train.y_enc
    X_test = dataset.test.X_enc.astype(float)
    # y_test = dataset.test.y_enc

    log.warning('ignoring runtime.')  # Not available? just number of iterations.
    log.warning('ignoring n_cores.')  # Not available
    log.warning('always optimize towards accuracy.')  # loss_fn lambda y1,y2:loss(y1, y2)
    hyperoptsklearn = HyperoptEstimator(classifier=any_classifier('clf'), algo=tpe.suggest)
    hyperoptsklearn.fit(X_train, y_train)
    class_predictions = hyperoptsklearn.predict(X_test)
    class_probabilities = one_hot_encode_predictions(class_predictions, dataset.target)

    dest_file = os.path.join(os.path.expanduser(config.output_dir), "predictions_decision_tree_{task}_{fold}.txt".format(task=config.name, fold=config.fold))
    save_predictions_to_file(class_probabilities, class_predictions, dest_file)
    log.info("Predictions saved to %s", dest_file)

