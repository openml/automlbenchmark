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
from frameworks.shared.callee import call_run, result, save_metadata

log = logging.getLogger(__name__)

def run(dataset: Dataset, config: TaskConfig):
    avaible_task_list = ['classification', 'regression']
    if config.type not in avaible_task_list:
        raise ValueError('{} is not supported.'.format(config.type))

    dir_path = os.path.dirname(os.path.realpath(__file__))
    DOTNET_INSTALL_DIR = os.path.join(dir_path, '.dotnet')
    os.environ['MODELBUILDER_AUTOML'] = 'NNI'
    os.environ['DOTNET_ROOT'] = DOTNET_INSTALL_DIR
    mlnet = os.path.join(DOTNET_INSTALL_DIR, 'mlnet')
    save_metadata(config)
    name = config.name
    temp_output_folder = os.path.join(config.output_dir,name)
    if not os.path.exists(temp_output_folder):
        os.mkdir(temp_output_folder)
    train_time_in_seconds = config.max_runtime_seconds
    log_path = os.path.join(temp_output_folder, 'log.txt')
    sub_command = config.type
    column_num = dataset.train.X.shape[1]
    columns=['column_{}'.format(i) for i in range(column_num)]
    train_df = pd.DataFrame(dataset.train.X, columns=columns)
    train_df['label'] = dataset.train.y
    test_df = pd.DataFrame(dataset.test.X, columns=columns)
    test_df['label'] = dataset.test.y
    train_dataset = os.path.join(temp_output_folder, 'train.csv')
    test_dataset = os.path.join(temp_output_folder, 'test.csv')
    log.info("saving train to {}".format(train_dataset))
    train_df.to_csv(train_dataset, index=False, header=True)
    test_df.to_csv(test_dataset, index=False, header=True)
    with Timer() as training:
        cmd = '{} {}'.format(mlnet, sub_command)

        # dataset & test dataset
        cmd += ' --dataset {} --test-dataset {}'.format(train_dataset, test_dataset)

        # train time
        cmd += ' --train-time {}'.format(train_time_in_seconds)

        # label
        cmd += ' --label-col label'

        # output folder & name
        cmd += ' --output {} --name {}'.format(config.output_dir, name)

        # log level & log file place
        cmd += ' --verbosity q --log-file-path {}'.format(log_path)
        run_cmd(cmd, _env_=os.environ)

        train_result_json = os.path.join(temp_output_folder, '{}.mbconfig'.format(name))
        if not os.path.exists(train_result_json):
            raise NoResultError("MLNet failed producing any prediction.")
        
    with open(train_result_json, 'r') as f:
        json_str = f.read()
        mb_config = json.loads(json_str)
        model_path = mb_config['Artifact']['MLNetModelPath']
        output_prediction_txt = config.output_predictions_file.replace('.csv', '.txt')
        # predict
        if config.type == 'classification':
            predict_cmd = '{} {}'.format(mlnet, 'predict')
            predict_cmd += ' --model {} --dataset {} --task-type classification'.format(model_path, test_dataset)
            predict_cmd += ' > {}'.format(output_prediction_txt)
            with Timer() as prediction:
                run_cmd(predict_cmd)
            prediction_df = pd.read_csv(output_prediction_txt, dtype={'PredictedLabel':'object'})
            #rename_df = prediction_df.rename(columns={'PredictedLabel':'predictions'})
            #rename_df['truth'] = dataset.test.y
            #rename_df.to_csv(config.output_predictions_file)
            save_predictions(
                dataset=dataset,
                output_file=config.output_predictions_file,
                predictions=prediction_df['PredictedLabel'].values,
                truth=dataset.test.y,
                probabilities=prediction_df.values[:,:-1],
                probabilities_labels=list(prediction_df.columns.values[:-1]),
            )

            return dict(
                training_duration=training.duration,
                predict_duration=prediction.duration,
            )
        
        if config.type == 'regression':
            predict_cmd = '{} {}'.format(mlnet, 'predict')
            predict_cmd += ' --model {} --dataset {} --task-type regression'.format(model_path, test_dataset)
            predict_cmd += ' > {}'.format(output_prediction_txt)
            with Timer() as prediction:
                run_cmd(predict_cmd)
            prediction_df = pd.read_csv(output_prediction_txt)
            rename_df = prediction_df.rename(columns={'Score':'predictions'})
            rename_df['truth'] = dataset.test.y
            rename_df.to_csv(config.output_predictions_file)

            return dict(
                training_duration=training.duration,
                predict_duration=prediction.duration,
            )
        


if __name__ == '__main__':
    call_run(run)
