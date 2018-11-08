import logging
import os

from sklearn.dummy import DummyClassifier

from automl.benchmark import TaskConfig
from automl.data import Dataset
from automl.utils import save_predictions_to_file


log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    log.info("\n**** Constant predictor (sklearn dummy) ****\n")

    classifier = DummyClassifier(strategy='prior')
    classifier.fit(dataset.train.X, dataset.train.y)
    class_probabilities = classifier.predict_proba(dataset.test.X)
    class_predictions = classifier.predict(dataset.test.X)

    save_predictions_to_file(class_probabilities, class_predictions, config.output_file_template)

