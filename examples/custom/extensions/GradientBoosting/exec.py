import logging

import sklearn
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.datautils import impute_array
from amlb.results import save_predictions
from amlb.utils import Timer

log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    log.info(f"\n**** Gradient Boosting [sklearn v{sklearn.__version__}] ****\n")

    is_classification = config.type == 'classification'

    X_train, X_test = impute_array(dataset.train.X_enc, dataset.test.X_enc)
    y_train, y_test = dataset.train.y, dataset.test.y

    estimator = GradientBoostingClassifier if is_classification else GradientBoostingRegressor
    predictor = estimator(random_state=config.seed, **config.framework_params)

    with Timer() as training:
        predictor.fit(X_train, y_train)
    predictions = predictor.predict(X_test)
    probabilities = predictor.predict_proba(X_test) if is_classification else None

    save_predictions(dataset=dataset,
                     output_file=config.output_predictions_file,
                     probabilities=probabilities,
                     predictions=predictions,
                     truth=y_test)

    return dict(
        models_count=1,
        training_duration=training.duration
    )
