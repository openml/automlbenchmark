# import standard_lib
import tempfile
import os
import logging
import psutil
import json

# import 3rd_parties
import pandas as pd
import numpy as np
import arff

# import amlb
from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.results import NoResultError, save_predictions
from amlb.utils import run_cmd, Timer
from frameworks.shared.callee import output_subdir, save_metadata

log = logging.getLogger(__name__)

def run(dataset: Dataset, config: TaskConfig):
    log.info(f"\n**** MLNet [v{config.framework_version}] ****\n")
    save_metadata(config)

    avaible_task_list = ['classification', 'regression']
    if config.type not in avaible_task_list:
        raise ValueError(f'{config.type} is not supported.')

    dir_path = os.path.dirname(os.path.realpath(__file__))
    DOTNET_INSTALL_DIR = os.path.join(dir_path, 'lib')
    os.environ['DOTNET_ROOT'] = DOTNET_INSTALL_DIR
    os.environ['MLNetCLIEnablePredict'] = 'True'
    threads_count_per_core = psutil.cpu_count() / psutil.cpu_count(logical=False)
    os.environ['MLNET_MAX_THREAD'] = str(config.cores * threads_count_per_core)
    mlnet = os.path.join(DOTNET_INSTALL_DIR, 'mlnet')
    train_time_in_seconds = config.max_runtime_seconds
    sub_command = config.type

    # set up MODELBUILDER_AUTOML
    MODELBUILDER_AUTOML = config.framework_params.get('automl_type', 'NNI')
    os.environ['MODELBUILDER_AUTOML'] = MODELBUILDER_AUTOML
    is_save_artifacts = config.framework_params.get('_save_artifacts', False)
    tmpdir = tempfile.mkdtemp()
    if is_save_artifacts:
        tmpdir = output_subdir('artifacts', config=config)

    train_dataset_path: str
    test_dataset_path: str
    label: str = 'label'
    temp_output_folder = os.path.join(tmpdir, str(config.fold))
    log_path = os.path.join(temp_output_folder, 'log.txt')
    label = dataset.target.index

    if dataset.train.format == 'csv':
        train_dataset_path = dataset.train.path
        test_dataset_path = dataset.test.path
    else:
        # .arff
        process = psutil.Process(os.getpid())
        log.info("before creating train/test dataframe")
        log.info(process.memory_info())
        train_dataset_path = os.path.join(tmpdir, f'train.csv')
        test_dataset_path = os.path.join(tmpdir, f'test.csv')
        columns = [f'col_{i}' for i in range(dataset.train.data.shape[-1])]
        columns[label] = 'label'

        # might use a bit extra memory, but it's faster
        pd.DataFrame(dataset.train.data, columns=columns).to_csv(train_dataset_path, index=None)
        pd.DataFrame(dataset.test.data, columns=columns).to_csv(test_dataset_path, index=None)

    log.info(f'train dataset: {train_dataset_path}')
    log.info(f'test dataset: {test_dataset_path}')
    
    cmd =   f"{mlnet} {sub_command}"\
            f" --dataset {train_dataset_path} --test-dataset {test_dataset_path} --train-time {train_time_in_seconds}"\
            f" --label-col {label} --output {os.path.dirname(temp_output_folder)} --name {config.fold}"\
            f" --verbosity q --log-file-path {log_path}"
    
    with Timer() as training:
        run_cmd(cmd)

    train_result_json = os.path.join(temp_output_folder, '{}.mbconfig'.format(config.fold))
    if not os.path.exists(train_result_json):
        raise NoResultError("MLNet failed producing any prediction.")
    
    with open(train_result_json, 'r') as f:
        json_str = f.read()
        mb_config = json.loads(json_str)
        model_path = mb_config['Artifact']['MLNetModelPath']
        output_prediction_txt = os.path.join(tmpdir, "prediction.txt")
        models_count = len(mb_config['RunHistory']['Trials'])
        # predict
        predict_cmd =   f"{mlnet} predict --task-type {config.type}" \
                        f" --model {model_path} --dataset {test_dataset_path} > {output_prediction_txt}"
        with Timer() as prediction:
            run_cmd(predict_cmd)
        if config.type == 'classification':
            prediction_df = pd.read_csv(output_prediction_txt, dtype={'PredictedLabel':'object'})
            save_predictions(
                dataset=dataset,
                output_file=config.output_predictions_file,
                predictions=prediction_df['PredictedLabel'].values,
                truth=dataset.test.y,
                probabilities=prediction_df.values[:,:-1],
                probabilities_labels=list(prediction_df.columns.values[:-1]),
            )
    
        if config.type == 'regression':
            prediction_df = pd.read_csv(output_prediction_txt)
            save_predictions(
                dataset=dataset,
                output_file=config.output_predictions_file,
                predictions=prediction_df['Score'].values,
                truth=dataset.test.y,
            )

        return dict(
                models_count = models_count,
                training_duration=training.duration,
                predict_duration=prediction.duration,
            )