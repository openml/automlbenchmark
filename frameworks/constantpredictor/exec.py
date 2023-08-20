import logging

import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor

from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.results import save_predictions
from amlb.utils import Timer, unsparsify
from frameworks.shared.callee import measure_inference_times

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
    log.info(f"Finished fit in {training.duration}s.")

    with Timer() as predict:
        predictions = predictor.predict(X_test)
    probabilities = predictor.predict_proba(X_test) if is_classification else None
    log.info(f"Finished predict in {predict.duration}s.")

    def infer(data):
        data = pd.read_parquet(data) if isinstance(data, str) else data
        return predictor.predict(data)

    inference_times = {}
    if config.measure_inference_time:
        inference_times["file"] = measure_inference_times(infer, dataset.inference_subsample_files(fmt="parquet"))
        test_data = X_test if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test)
        inference_times["df"] = measure_inference_times(
            infer,
            [(1, test_data.sample(1, random_state=i)) for i in range(100)],
        )
        log.info(f"Finished inference time measurements.")

    save_predictions(dataset=dataset,
                     output_file=config.output_predictions_file,
                     probabilities=probabilities,
                     predictions=predictions,
                     truth=y_test,
                     target_is_encoded=encode)

    return dict(
        models_count=1,
        training_duration=training.duration,
        predict_duration=predict.duration,
        inference_times=inference_times,
    )
