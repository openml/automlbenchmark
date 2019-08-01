"""
This script finds the output files for each benchmark run, and processes them for upload to OpenML.
The script requires the following variables, which currently are just hard-coded.
"""
from collections import OrderedDict, defaultdict
import os
import re
from typing import Iterable, Tuple, Dict, List, Any
import warnings

import openml
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

# a local directory with all files from the automlbenchmark s3 bucket.
local_bucket_directory: str = r'D:\data\amlb\ec2'
# we assume the following structure within the directory:
directory_format = (r".+\\{mode}_{framework}_{benchmark}_.+"  # the .+ are resp. path prefix and a UTC timestamp
                    r"\\{mode}_{benchmark}_{taskname}_{fold}_{framework}")
# regex match is a bit weak as its an internal script. It suffices and does not need to work against adversaries.
mode = 'aws'

# a file from the reports directory for which to find the original prediction files
results_to_match: str = r'..\reports\results_medium-8c4h.csv'

amlb_flows = dict(
    autosklearn=15509,
    h2oautoml=16115,
    tpot=16114,
    autoweka=16116,
    randomforest=16118,
    tunedrandomforest=16119,
)


def find_output_directories(framework: str, task: str, fold: str, benchmark: str) -> Iterable[str]:
    """ Finds the directories which might contain predictions for the specified arguments.

    :param framework: str.
        Name as in /resources/frameworks.yaml (or lower case)
    :param task: str
        The name of the task as it is known on OpenML. Refer to the names in /resources/benchmarks/XYZ.yaml
    :param fold: str
        The fold number.
    :param benchmark: str
        The benchmark name (e.g. 'large-8c4h')
    :return:
        An iterable with all directories which could contain the used predictions.
    """
    framework = framework.lower()
    pattern = directory_format.format(mode=mode, framework=framework, benchmark=benchmark, taskname=task, fold=fold)
    for subdirectory in os.listdir(local_bucket_directory):
        for subsubdirectory in os.listdir(os.path.join(local_bucket_directory, subdirectory)):
            path = os.path.join(local_bucket_directory, subdirectory, subsubdirectory)
            if re.match(pattern, path):
                yield path


# aws_autosklearn_large-8c4h_20190418T123448
def find_predictions(benchmark: str, result_row: pd.Series) -> str:
    """ Finds the predictions file for the (framework, task, fold)-tuple.

    :param benchmark: str
        The benchmark name (e.g. 'large-8c4h')
    :param result_row: pd.Series
        A row in from the `results_to_match` file for which to find the prediction file.
    :return:
        The path to the prediction file.
    """
    for output_directory in find_output_directories(row.framework, row.task, row.fold, benchmark):
        expected_prediction_file = f'output/predictions/{row.framework}_{row.task}_{row.fold}.csv'
        full_prediction_path = os.path.join(output_directory, expected_prediction_file)
        results_path = os.path.join(output_directory, 'output/results.csv')
        if os.path.isfile(full_prediction_path) and os.path.isfile(results_path):
            results = pd.read_csv(results_path)
            if len(results) > 1:
                raise RuntimeError(f"{results_path} contains more than one row.")
            results = results.iloc[0]
            if result_row.result == results.result:
                return full_prediction_path
    else:
        warnings.warn(f"No match found for {benchmark} + {row.framework} + {row.task} + {row.fold}!")


def parse_resource_parameters(benchmark_string: str) -> Tuple[str, str, str]:
    """ Parses the benchmark name string to find the number of cores and amount of time allowed. """
    # benchmark_string is e.g. 'medium-8c4h'
    _, parameters = benchmark_string.split('-')
    n_cores, remainder = parameters.split('c')
    time = remainder[:-1]
    memory = '32'  # Currently all experiments are run on 32Gb machines
    return n_cores, memory, time


def load_format_predictions(task_id: int, predictions: Dict[int, str]) -> List[List[Any]]:
    """ Converts the benchmark-stored predictions to a format that can be uploaded to OpenML.

    :param task_id: task for which predictions are done
    :param predictions: dict mapping each fold to a predictions file
    :return: openml-compatible formatted predictions:
        [[repeat, fold, sample, index, p1, ..., pk, y_pred, y_true], ...]
        where p1, ..., pk are predicted class probabilities
              y_pred is the predicted label, y_true is the  true label
    """
    # In the benchmark prediction files, predictions are stored per fold as:
    # p1, ..., pk, y_pred, y_true
    # No repeats or samples are used. We only need to find matching indices for the task
    task = openml.tasks.get_task(task_id)
    assert (1, 10, 1) == task.get_split_dimensions()  # check repeats/folds/samples are as expected

    formatted_predictions = []
    for fold in range(10):
        fold_predictions = pd.read_csv(predictions[fold])
        # for non-numeric class labels, it seems that we store the class probabilities in a different order
        # so we re-arrange our probability predictions to match what OpenML expects:
        class_names = fold_predictions.columns[:-2]
        # stop of class names don't match
        if set(task.class_labels) != set(class_names):
            raise RuntimeError(f"For task {task_id} the labels differ. "
                               f"Expected: {task.class_labels}, actual: {class_names}")
        # if class names match, ensure that the ordering is also the same
        if all(k1 != k2 for (k1, k2) in zip(task.class_labels, class_names)):
            fold_predictions = fold_predictions[task.class_labels + list(fold_predictions.columns[-2:])]

        train, test = task.get_train_test_split_indices(fold, repeat=0, sample=0)
        assert len(fold_predictions) == len(test), (
            f"Number of predictions not equal to length of test set for fold {fold} of task {task_id}.")
        for index, predicted in zip(test, fold_predictions.values):
            formatted_predictions.append(
                [0, fold, 0, index, *predicted]
            )

    return formatted_predictions


def create_run(benchmark: str, framework: str, task_id: int, predictions: Dict[int, str]) -> openml.runs.OpenMLRun:
    """
    :param benchmark: benchmark name containing allowed resources, e.g. 'medium-8c4h'
    :param framework: framework name
    :param task_id: openml task id
    :param predictions: mapping for fold->predictions file
    :return: an OpenML run connected between the right task and flow, and associated predictions.
    """
    cores, memory, time = parse_resource_parameters(benchmark)
    flow_id = amlb_flows[framework]

    parameters = [
        OrderedDict([('oml:name', 'cores'), ('oml:value', cores), ('oml:component', flow_id)]),
        OrderedDict([('oml:name', 'memory'), ('oml:value', memory), ('oml:component', flow_id)]),
        OrderedDict([('oml:name', 'time'), ('oml:value', time), ('oml:component', flow_id)]),
    ]

    task = openml.tasks.get_task(task_id)
    dataset_id = task.get_dataset().dataset_id

    benchmark_command = f'python3 runbenchmark.py {framework} {benchmark} -m aws -t {task_id}'

    predictions = load_format_predictions(task_id, predictions)

    return openml.runs.OpenMLRun(
        task_id=task_id, flow_id=flow_id, dataset_id=dataset_id,
        parameter_settings=parameters,
        setup_string=benchmark_command,
        data_content=predictions,
        tags=['study_218']
    )


def compare_results(results: pd.DataFrame, run: openml.runs.OpenMLRun, framework: str, task: str):
    class MockExtension():
        def get_version_information(self):
            return ['mock']

    run.flow.extesion = MockExtension()
    run.get_metric_fn(accuracy_score)


# Wrong D:\\data\\amlb\\ec2\\aws_autosklearn_medium-8c4h_20190415T183932\\aws_medium-8c4h_adult_2_autosklearn
# Right D:\\data\\amlb\\ec2\\aws_autosklearn_medium-8c4h_20190424T204255\\aws_medium-8c4h_adult_2_autosklearn
if __name__ == '__main__':
    results = pd.read_csv(results_to_match)
    results.framework = results.framework.str.lower()
    results.task = results.task.str.lower()

    # We can ignore frameworks not used in the benchmark paper
    results = results[results.framework != 'oboe']

    benchmark = 'medium-8c4h'
    # keeps mapping of (framework, task) -> (fold -> predictions file)
    predictions_mapping: Dict[Tuple[str, int], Dict[int, str]] = defaultdict(dict)
    for i, row in results.iterrows():
        task_id = int(row.id.split('/')[-1])
        predictions_mapping[(row.framework, task_id)][row.fold] = find_predictions(benchmark, row)

    uploaded_tasks = [34539,
    168868,
    14965,
    146195,
    146825,
    168337,
    168329,
    146606,
    168330,
    167119,
    3945,
    168335,
    9977,
    167120,
    168338
    ]
    for (framework, task), predictions in predictions_mapping.items():
        if len(predictions) != 10:
            print(f'Task {task} does not have predictions for 10 folds (has {predictions.items()}).')
        else:
            if framework == 'autosklearn' and task in uploaded_tasks:
                continue
            run = create_run(benchmark, framework, task, predictions)
            print(f'{framework} {task}: {run.publish().run_id}')
