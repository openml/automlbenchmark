import logging
import os
from sklearn.tree import DecisionTreeClassifier

from automl.benchmark import TaskConfig
from automl.data import Dataset
from automl.utils import save_predictions_to_file

log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    log.info("\n**** Decision Tree (sklearn) ****\n")

    classifier = DecisionTreeClassifier()
    classifier.fit(dataset.train.X, dataset.train.y)
    class_probabilities = classifier.predict_proba(dataset.test.X)
    class_predictions = classifier.predict(dataset.test.X)

    dest_file = os.path.join(os.path.expanduser(config.output_folder), "predictions_decision_tree_{task}_{fold}.txt".format(task=config.name, fold=config.fold))
    save_predictions_to_file(class_probabilities, class_predictions, dest_file)
    log.info("Predictions saved to %s", dest_file)

