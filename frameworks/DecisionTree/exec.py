import logging

import sklearn
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.datautils import impute_array
from amlb.results import save_predictions
from amlb.utils import Timer, unsparsify

log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    log.info(f"\n**** Decision Tree [sklearn v{sklearn.__version__}] ****\n")

    is_classification = config.type == 'classification'

    X_train, X_test = impute_array(*unsparsify(dataset.train.X_enc, dataset.test.X_enc, fmt='array'))
    y_train, y_test = unsparsify(dataset.train.y_enc, dataset.test.y_enc, fmt='array')

    estimator = DecisionTreeClassifier if is_classification else DecisionTreeRegressor
    predictor = estimator(random_state=config.seed, **config.framework_params)

    with Timer() as training:
        predictor.fit(X_train, y_train)
    log.info(f"Finished fit in {training.duration}s.")

    with Timer() as predict:
        predictions = predictor.predict(X_test)
    probabilities = predictor.predict_proba(X_test) if is_classification else None
    log.info(f"Finished predict in {predict.duration}s.")

    save_predictions(dataset=dataset,
                     output_file=config.output_predictions_file,
                     probabilities=probabilities,
                     predictions=predictions,
                     truth=y_test,
                     target_is_encoded=is_classification)

    return dict(
        models_count=1,
        training_duration=training.duration,
        predict_duration=predict.duration
    )
