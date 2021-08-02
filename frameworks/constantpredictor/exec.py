import logging

from sklearn.dummy import DummyClassifier, DummyRegressor

from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.results import save_predictions
from amlb.utils import Timer, unsparsify

log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    log.info("\n**** Constant predictor (sklearn dummy) ****\n")

    is_classification = config.type == 'classification'
    predictor = DummyClassifier(strategy='prior') if is_classification else DummyRegressor(strategy='median')

    encode = config.framework_params.get('_encode', False)

    X_train = unsparsify(dataset.train.X_enc if encode else dataset.train.X, fmt='array')
    y_train = unsparsify(dataset.train.y_enc if encode else dataset.train.y, fmt='array')
    X_test = unsparsify(dataset.test.X_enc if encode else dataset.test.X, fmt='array')
    y_test = unsparsify(dataset.test.y_enc if encode else dataset.test.y, fmt='array')

    with Timer() as training:
        predictor.fit(X_train, y_train)
    with Timer() as predict:
        predictions = predictor.predict(X_test)
    probabilities = predictor.predict_proba(X_test) if is_classification else None

    save_predictions(dataset=dataset,
                     output_file=config.output_predictions_file,
                     probabilities=probabilities,
                     predictions=predictions,
                     truth=y_test,
                     target_is_encoded=encode)

    return dict(
        models_count=1,
        training_duration=training.duration,
        predict_duration=predict.duration
    )
