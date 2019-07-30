"""
This script finds the output files for each benchmark run, and processes them for upload to OpenML.
The script requires the following variables, which currently are just hard-coded.
"""
import os
import re
from typing import Iterable
import warnings
import pandas as pd

# a local directory with all files from the automlbenchmark s3 bucket.
local_bucket_directory: str = r'D:\data\amlb\ec2'
# we assume the following structure within the directory:
directory_format = (r".+\\{mode}_{framework}_{benchmark}_.+"  # the .+ are resp. path prefix and a UTC timestamp
                    r"\\{mode}_{benchmark}_{taskname}_{fold}_{framework}")
# regex match is a bit weak as its an internal script. It suffices and does not need to work against adversaries.
mode = 'aws'

# a file from the reports directory for which to find the original prediction files
results_to_match: str = r'..\reports\results_medium-8c4h.csv'


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


# Wrong D:\\data\\amlb\\ec2\\aws_autosklearn_medium-8c4h_20190415T183932\\aws_medium-8c4h_adult_2_autosklearn
# Right D:\\data\\amlb\\ec2\\aws_autosklearn_medium-8c4h_20190424T204255\\aws_medium-8c4h_adult_2_autosklearn
if __name__ == '__main__':
    results = pd.read_csv(results_to_match)
    results.framework = results.framework.str.lower()
    results.task = results.task.str.lower()

    # We can ignore frameworks not used in the benchmark paper
    results = results[results.framework != 'oboe']

    for i, row in results.iterrows():
        print(find_predictions('medium-8c4h', row))

    #print(list(find_output_directories('autosklearn', 'adult', '2', 'medium-8c4h')))
