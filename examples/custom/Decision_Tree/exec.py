import logging
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from automl.benchmark import TaskConfig
from automl.data import Dataset
from automl.datautils import impute
from automl.results import save_predictions_to_file
from automl.utils import Timer

log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    log.info("\n**** Decision Tree (sklearn) ****\n")

    is_classification = config.type == 'classification'

    X_train, X_test = impute(dataset.train.X_enc, dataset.test.X_enc)
    y_train, y_test = dataset.train.y, dataset.test.y

    estimator = DecisionTreeClassifier if is_classification else DecisionTreeRegressor
    predictor = estimator(random_state=config.seed, **config.framework_params)

    with Timer() as training:
        predictor.fit(X_train, y_train)
    predictions = predictor.predict(X_test)
    probabilities = predictor.predict_proba(X_test) if is_classification else None

    save_predictions_to_file(dataset=dataset,
                             output_file=config.output_predictions_file,
                             probabilities=probabilities,
                             predictions=predictions,
                             truth=y_test)

    return dict(
        models_count=1,
        training_duration=training.duration
    )
