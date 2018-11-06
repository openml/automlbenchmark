import os
from sklearn.dummy import DummyClassifier

from automl.benchmark import TaskConfig
from automl.data import Dataset
from automl.utils import save_predictions_to_file


def run(dataset: Dataset, config: TaskConfig):
    print("\n**** Constant predictor (sklearn dummy) ****\n")

    classifier = DummyClassifier(strategy='prior')
    classifier.fit(dataset.train.X, dataset.train.y)
    class_probabilities = classifier.predict_proba(dataset.test.X)
    class_predictions = classifier.predict(dataset.test.X)

    dest_file = os.path.join(os.path.expanduser(config.output_folder), "predictions_random_forest_{task}_{fold}.txt".format(task=config.name, fold=config.fold))
    save_predictions_to_file(class_probabilities, class_predictions, dest_file)
    print("Predictions saved to "+dest_file)
    print()

