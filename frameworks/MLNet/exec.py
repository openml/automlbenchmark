import logging
import math
import os
import pandas as pd
from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.datautils import reorder_dataset
from amlb.results import NoResultError, save_predictions
from amlb.utils import dir_of, path_from_split, run_cmd, split_path, Timer
import json
from frameworks.shared.callee import call_run, result, save_metadata,  output_subdir

log = logging.getLogger(__name__)

def run(dataset: Dataset, config: TaskConfig):
    log.info(f"\n**** MLNet [v{config.framework_version}] ****\n")
    save_metadata(config)

    avaible_task_list = ['classification', 'regression']
    if config.type not in avaible_task_list:
        raise ValueError(f'{config.type} is not supported.')

    dir_path = os.path.dirname(os.path.realpath(__file__))
    DOTNET_INSTALL_DIR = os.path.join(dir_path, 'lib')
    os.environ['MODELBUILDER_AUTOML'] = 'NNI'
    os.environ['DOTNET_ROOT'] = DOTNET_INSTALL_DIR
    mlnet = os.path.join(DOTNET_INSTALL_DIR, 'mlnet')
    temp_output_folder = output_subdir('', config=config)
    train_time_in_seconds = config.max_runtime_seconds
    log_path = os.path.join(temp_output_folder, 'log.txt')
    sub_command = config.type
    train_dataset = os.path.join(config.output_dir, f'train_{config.fold}.csv')
    test_dataset = os.path.join(config.output_dir, f'test_{config.fold}.csv')
    if not os.path.exists(train_dataset) or not os.path.exists(test_dataset):
        column_num = dataset.train.X.shape[1]
        columns=[f'column_{i}' for i in range(column_num)]
        train_df = pd.DataFrame(dataset.train.X, columns=columns)
        train_df['label'] = dataset.train.y
        test_df = pd.DataFrame(dataset.test.X, columns=columns)
        test_df['label'] = dataset.test.y

        log.info(f'saving train to {train_dataset}')
        train_df.to_csv(train_dataset, index=False, header=True)
        log.info(f'saving test to {test_dataset}')
        test_df.to_csv(test_dataset, index=False, header=True)

    with Timer() as training:
        cmd =   f"{mlnet} {sub_command}"\
                f" --dataset {train_dataset} --test-dataset {test_dataset} --train-time {train_time_in_seconds}"\
                f" --label-col label --output {os.path.dirname(temp_output_folder)} --name {config.fold}"\
                f" --verbosity q --log-file-path {log_path}"
        run_cmd(cmd)

        train_result_json = os.path.join(temp_output_folder, '{}.mbconfig'.format(config.fold))
        if not os.path.exists(train_result_json):
            raise NoResultError("MLNet failed producing any prediction.")
        
    with open(train_result_json, 'r') as f:
        json_str = f.read()
        mb_config = json.loads(json_str)
        model_path = mb_config['Artifact']['MLNetModelPath']
        output_prediction_txt = config.output_predictions_file.replace('.csv', '.txt')
        models_count = len(mb_config['RunHistory']['Trials'])

        # predict
        predict_cmd =   f"{mlnet} predict --task-type {config.type}" \
                        f" --model {model_path} --dataset {test_dataset} > {output_prediction_txt}"
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
