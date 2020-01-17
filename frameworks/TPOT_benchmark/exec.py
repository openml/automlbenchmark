import logging

import numpy as np

from autogluon_utils.benchmarking.baselines.tpot_base.tpot_base import TPOTBaseline
from autogluon_utils.benchmarking.openml.automlbenchmark_wrapper import prepare_data

from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.results import save_predictions_to_file
from amlb.utils import Timer

log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    log.info("\n**** TPOT Benchmark ****\n")

    X_train, y_train, X_test, y_test, problem_type, perf_metric = prepare_data(config=config, dataset=dataset)

    X_train['__label__'] = y_train
    baseline = TPOTBaseline()
    with Timer() as training:
        num_models_trained, num_models_ensemble, fit_time = baseline.fit(
            train_data=X_train,
            label_column='__label__',
            problem_type=problem_type,
            output_directory='tmp/',
            eval_metric=perf_metric,
            runtime_sec=config.max_runtime_seconds,
            # runtime_sec=60,
            random_state=0,
            num_cores=config.cores,
        )

    is_classification = config.type == 'classification'
    if is_classification:
        predictions, probabilities, predict_time = baseline.predict(test_data=X_test, pred_class_and_proba=True)
    else:
        predictions, probabilities, predict_time = baseline.predict(test_data=X_test)
    print('timer time:', training.duration)
    print('baseline time:', fit_time)
    print('predict time:', predict_time)
    print('num_models_trained:', num_models_trained)
    print('num_models_ensemble:', num_models_ensemble)

    if is_classification & (len(probabilities.shape) == 1):
        probabilities = np.array([[1-row, row] for row in probabilities])

    save_predictions_to_file(dataset=dataset,
                             output_file=config.output_predictions_file,
                             probabilities=probabilities,
                             predictions=predictions,
                             truth=y_test,
                             target_is_encoded=False)

    return dict(
        models_count=num_models_trained,
        training_duration=training.duration
    )
