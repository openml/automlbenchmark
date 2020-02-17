import logging

import numpy as np
import boto3

from autogluon_utils.benchmarking.baselines.autopilot.autopilot_base import AutoPilotBaseline
from autogluon_utils.benchmarking.baselines.autopilot.autopilot_aws_config import DEFAULT_AUTOPILOT_CONFIG
from autogluon_utils.benchmarking.openml.automlbenchmark_wrapper import prepare_data

from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.results import save_predictions_to_file
from amlb.utils import Timer

log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    log.info("\n**** AutoPilot Benchmark ****\n")

    if config.fold != 0:
        raise AssertionError('config.fold must be 0 when running AutoPilot! Value: %s' % config.fold)

    X_train, y_train, X_test, y_test, problem_type, perf_metric = prepare_data(config=config, dataset=dataset)

    if DEFAULT_AUTOPILOT_CONFIG['region_name'] == 'YOUR_REGION':
        raise AssertionError('To run AutoPilot, region_name, s3_bucket, and role_arn have to be specified in the code')

    session = boto3.session.Session(region_name=DEFAULT_AUTOPILOT_CONFIG['region_name'])
    s3_bucket = DEFAULT_AUTOPILOT_CONFIG['s3_bucket']
    role_arn = DEFAULT_AUTOPILOT_CONFIG['role_arn']

    dataset_name_suffix = '-amlb-' + str(config.max_runtime_seconds) + '-f' + str(config.fold)
    num_chars_left = 32 - 4 - len(dataset_name_suffix)
    dataset_name_prefix = config.name[:num_chars_left]
    dataset_name = dataset_name_prefix + dataset_name_suffix
    dataset_name = dataset_name.replace('.', '-')
    dataset_name = dataset_name.replace('_', '-')
    job_id = dataset_name

    X_train['__label__'] = y_train
    baseline = AutoPilotBaseline()
    with Timer() as training:
        num_models_trained, num_models_ensemble, fit_time = baseline.fit(
            train_data=X_train,
            label_column='__label__',
            problem_type=problem_type,
            session=session,
            job_id=job_id,
            s3_bucket=s3_bucket,
            role_arn=role_arn,
            eval_metric=perf_metric.name,
            runtime_sec=config.max_runtime_seconds,
        )

    is_classification = config.type == 'classification'
    predictions, probabilities, predict_time = baseline.predict(test_data=X_test, pred_class_and_proba=True)

    print('timer time:', training.duration)
    print('baseline time:', fit_time)
    print('predict time:', predict_time)
    print('num_models_trained:', num_models_trained)
    print('num_models_ensemble:', num_models_ensemble)

    classes = baseline.classes

    if (is_classification) & (len(probabilities.shape) == 1):
        probabilities = np.array([[1-row, row] for row in probabilities])

    if is_classification:
        print(classes)
        print(predictions[:5])
        print(probabilities[:5])
        print(y_test[:5])

    save_predictions_to_file(dataset=dataset,
                             output_file=config.output_predictions_file,
                             probabilities=probabilities,
                             predictions=predictions,
                             truth=y_test,
                             target_is_encoded=False,
                             probabilities_labels=classes)

    return dict(
        models_count=num_models_trained,
        models_ensemble_count=num_models_ensemble,
        training_duration=training.duration,
        predict_duration=predict_time,
    )
