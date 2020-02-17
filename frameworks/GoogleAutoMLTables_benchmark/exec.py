import logging

import numpy as np

from autogluon_utils.benchmarking.baselines.gcp.methods_gcp import gcptables_fit_predict
from autogluon_utils.benchmarking.openml.automlbenchmark_wrapper import prepare_data

from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.results import save_predictions_to_file
from amlb.utils import Timer

log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    log.info("\n**** Google AutoML Tables Benchmark ****\n")

    if config.fold != 0:
        raise AssertionError('config.fold must be 0 when running AutoML Tables! Value: %s' % config.fold)

    X_train, y_train, X_test, y_test, problem_type, perf_metric = prepare_data(config=config, dataset=dataset)

    gcp_info = {  # GCP configuration info. You should only change: GOOGLE_APPLICATION_CREDENTIALS to location of GCP key file.
        'PROJECT_ID': "YOUR_PROJECT_ID",
        'BUCKET_NAME': "YOUR_BUCKET_NAME",
        'COMPUTE_REGION': "us-central1",
        'GOOGLE_APPLICATION_CREDENTIALS': 'PATH_TO_YOUR_CREDENTIALS_FILE',
    }
    if gcp_info['GOOGLE_APPLICATION_CREDENTIALS'] == 'PATH_TO_YOUR_CREDENTIALS_FILE':
        raise AssertionError('Edit source code and add your Google account credentials to run GCP-Tables')

    # TODO: RENAME MODEL BASED ON TIME, OTHERWISE IT WON'T BE UNIQUE ACROSS 1hr and 4hr!
    X_train['googleautomllabel'] = y_train

    # TODO: Replace name prefix - with _, and any other specials like .
    dataset_name_gcp_suffix = '_amlb_' + str(config.max_runtime_seconds) + '_f' + str(config.fold)
    num_chars_left = 32 - 4 - len(dataset_name_gcp_suffix)
    dataset_name_gcp_prefix = config.name[:num_chars_left]
    dataset_name_gcp = dataset_name_gcp_prefix + dataset_name_gcp_suffix
    dataset_name_gcp = dataset_name_gcp.replace('.', '_')
    dataset_name_gcp = dataset_name_gcp.replace('-', '_')
    model_name = 'tm_' + dataset_name_gcp
    with Timer() as training:
        num_models_trained, num_models_ensemble, fit_time, predictions, probabilities, predict_time, class_order = gcptables_fit_predict(
            train_data=X_train,
            test_data=X_test,
            dataset_name=dataset_name_gcp,
            label_column='googleautomllabel',
            problem_type=problem_type,
            output_directory='tmp',
            gcp_info=gcp_info,
            eval_metric=perf_metric.name,
            runtime_sec=config.max_runtime_seconds,
            fit_model=True,
            model_name=model_name,
            make_predictions=True,
        )

    print('timer time:', training.duration)
    print('baseline time:', fit_time)
    print('predict time:', predict_time)
    print('num_models_trained:', num_models_trained)
    print('num_models_ensemble:', num_models_ensemble)

    is_classification = config.type == 'classification'
    if is_classification & (len(probabilities.shape) == 1):
        probabilities = np.array([[1-row, row] for row in probabilities])

    classes = class_order

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
