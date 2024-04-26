# import standard_lib
import logging
import json
import os
import psutil
import shutil
import tempfile

# import 3rd_parties
import pandas as pd

# import amlb
from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.results import NoResultError, save_predictions
from amlb.utils import clean_dir, run_cmd, zip_path, Timer
from frameworks.shared.callee import output_subdir

log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    log.info(f"\n**** MLNet [v{config.framework_version}] ****\n")

    avaible_task_list = ['classification', 'regression']
    if config.type not in avaible_task_list:
        raise ValueError(f'{config.type} is not supported.')

    dir_path = os.path.dirname(os.path.realpath(__file__))
    DOTNET_INSTALL_DIR = os.path.join(dir_path, 'lib')
    os.environ['DOTNET_ROOT'] = DOTNET_INSTALL_DIR
    os.environ['MLNetCLIEnablePredict'] = 'True'
    os.environ['MLNET_MAX_THREAD'] = str(config.cores)
    mlnet = os.path.join(DOTNET_INSTALL_DIR, 'mlnet')
    train_time_in_seconds = config.max_runtime_seconds
    sub_command = config.type

    # Opt-out of Telemetry via framework parameter
    MLDOTNET_CLI_TELEMETRY_OPTOUT = config.framework_params.get('_telemetry_disabled', 'False')
    os.environ['MLDOTNET_CLI_TELEMETRY_OPTOUT'] = MLDOTNET_CLI_TELEMETRY_OPTOUT

    # set up MODELBUILDER_AUTOML
    MODELBUILDER_AUTOML = config.framework_params.get('automl_type', 'NNI')
    os.environ['MODELBUILDER_AUTOML'] = MODELBUILDER_AUTOML

    artifacts = config.framework_params.get('_save_artifacts', [])
    tmpdir = tempfile.mkdtemp()
    tmp_output_folder = os.path.join(tmpdir, str(config.fold))
    output_dir = output_subdir('models', config=config) if 'models' in artifacts else tmp_output_folder
    log_dir = output_subdir('logs', config=config) if 'logs' in artifacts else tmp_output_folder
    log_path = os.path.join(log_dir, 'log.txt')

    try:
        label = dataset.target.index
        train_dataset_path = dataset.train.data_path('csv')
        test_dataset_path = dataset.test.data_path('csv')

        log.info(f'train dataset: {train_dataset_path}')
        log.info(f'test dataset: {test_dataset_path}')

        cmd = (f"{mlnet} {sub_command}"
               f" --dataset {train_dataset_path} --test-dataset {test_dataset_path} --train-time {train_time_in_seconds}"
               f" --label-col {label} --output {os.path.dirname(output_dir)} --name {config.fold}"
               f" --verbosity q --log-file-path {log_path}")

        with Timer() as training:
            run_cmd(cmd)
        log.info(f"Finished fit in {training.duration}s.")

        train_result_json = os.path.join(output_dir, '{}.mbconfig'.format(config.fold))
        if not os.path.exists(train_result_json):
            raise NoResultError("MLNet failed producing any prediction.")

        with open(train_result_json, 'r') as f:
            json_str = f.read()
            mb_config = json.loads(json_str)
            model_path = os.path.join(output_dir, f"{config.fold}.zip")
            output_prediction_path = os.path.join(log_dir, "prediction.txt")  # keeping this in log dir as it contains useful error when prediction fails
            models_count = len(mb_config['RunHistory']['Trials'])
            # predict
            predict_cmd = (f"{mlnet} predict --task-type {config.type}"
                           f" --model {model_path} --dataset {test_dataset_path} --label-col {dataset.target.name} > {output_prediction_path}")
            with Timer() as predict:
                run_cmd(predict_cmd)
            log.info(f"Finished predict in {predict.duration}s.")
            if config.type == 'classification':
                prediction_df = pd.read_csv(output_prediction_path, dtype={'PredictedLabel': 'object'})

                save_predictions(
                    dataset=dataset,
                    output_file=config.output_predictions_file,
                    predictions=prediction_df['PredictedLabel'].values,
                    truth=dataset.test.y,
                    probabilities=prediction_df.values[:,:-1],
                    probabilities_labels=list(prediction_df.columns.values[:-1]),
                )

            if config.type == 'regression':
                prediction_df = pd.read_csv(output_prediction_path)
                save_predictions(
                    dataset=dataset,
                    output_file=config.output_predictions_file,
                    predictions=prediction_df['Score'].values,
                    truth=dataset.test.y,
                )

            return dict(
                    models_count=models_count,
                    training_duration=training.duration,
                    predict_duration=predict.duration,
                )
    finally:
        if 'logs' in artifacts:
            logs_zip = os.path.join(log_dir, "logs.zip")
            zip_path(log_dir, logs_zip)
            clean_dir(log_dir, filter_=lambda p: p != logs_zip)
        if 'models' in artifacts:
            models_zip = os.path.join(output_dir, "models.zip")
            zip_path(output_dir, models_zip)
            clean_dir(output_dir, filter_=lambda p: p != models_zip)

        shutil.rmtree(tmpdir, ignore_errors=True)
