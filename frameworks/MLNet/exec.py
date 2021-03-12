import logging
import math
import os
import pandas as pd
import subprocess
from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.datautils import reorder_dataset
from amlb.results import NoResultError, save_predictions
from amlb.utils import dir_of, path_from_split, run_cmd, split_path, Timer
import json
from frameworks.shared.callee import save_metadata

log = logging.getLogger(__name__)
os.environ['ModelBuilder.AutoMLType'] = 'NNI'


def run(dataset: Dataset, config: TaskConfig, mlnet):
    log.info(f"\n**** MLNet [v{config.framework_version}]****\n")
    temp_output_folder = config.output_dir
    train_time_in_seconds = config.max_runtime_seconds
    name = config.name
    log_path = os.path.join(temp_output_folder, name, 'log.txt')
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
        cmd += ' --output {} --name {}'.format(temp_output_folder, name)

        # log level & log file place
        cmd += ' --verbosity q --log-file-path {}'.format(log_path)
        run_cmd(cmd)

        train_result_json = os.path.join(temp_output_folder, name, '{}.mbconfig'.format(name))
        if not os.path.exists(train_result_json):
            raise NoResultError("MLNet failed producing any prediction.")
        
        with open(train_result_json, 'r') as f:
            json_str = f.read()
            mb_config = json.loads(json_str)
            metric = mb_config['RunHistory']['EvaluationMetrics']
            log.info(metric)

            return dict(
                training_duration=training.duration
            )




