import logging
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from automl.benchmark import TaskConfig
from automl.data import Dataset
from automl.results import save_predictions_to_file

log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    log.info("\n**** Decision Tree (sklearn) ****\n")

    is_classification = config.type == 'classification'

    estimator = DecisionTreeClassifier if is_classification else DecisionTreeRegressor
    predictor = estimator(**config.framework_params)

    predictor.fit(dataset.train.X, dataset.train.y)
    class_predictions =predictor.predict(dataset.test.X)
    class_probabilities = predictor.predict_proba(dataset.test.X) if is_classification else None

    save_predictions_to_file(dataset=dataset,
                             output_file=config.output_predictions_file,
                             class_probabilities=class_probabilities,
                             class_predictions=class_predictions,
                             class_truth=dataset.test.y)

