import json
import logging
import os
from collections import OrderedDict
from typing import Set, Optional, Tuple, List

import openml
import pandas as pd
from openml import OpenMLTask

from .utils.core import Namespace
from .__version__ import __version__


log = logging.getLogger(__name__)


def _load_fold(task_folder: str, fold: int, task: OpenMLTask) -> pd.DataFrame:
    """ Load the predictions and add openml repeat/fold/index information. """
    prediction_file = os.path.join(task_folder, f"{fold}/predictions.csv")
    predictions = pd.read_csv(prediction_file, sep=',', header=0)

    train_indices, test_indices = task.get_train_test_split_indices(fold, repeat=0, sample=0)
    predictions["index"] = test_indices
    predictions["fold"] = fold
    predictions["repeat"] = 0
    return predictions


def _load_predictions(task_folder: str) -> pd.DataFrame:
    """ Loads predictions of all folds for a task with index information required for upload. """
    metadata = _load_task_data(task_folder)
    task = openml.tasks.get_task(metadata.openml_task_id)
    results = [_load_fold(task_folder, fold, task) for fold in range(10)]
    return pd.concat(results)


def _load_task_data(task_folder: str, fold: int = 0) -> Namespace:
    """ Loads the metadata of the given fold of a task as a namespace. """
    with open(os.path.join(task_folder, str(fold), 'metadata.json'), 'r') as fh:
        metadata = json.load(fh)
    metadata = Namespace.from_dict(metadata)
    return metadata


# def _list_completed_folds(task_folder: str) -> Set[str]:
#     completed_folds = set()
#     for fold_dir in os.listdir(task_folder):
#         if "predictions.csv" in os.listdir(os.path.join(task_folder, fold_dir)):
#             completed_folds.add(fold_dir)
#     return completed_folds
#
#
def _get_flow(metadata: Namespace) -> openml.flows.OpenMLFlow:
    amlb_flow = openml.flows.OpenMLFlow(
        name=f"amlb_{metadata.framework}",
        description=f'{metadata.framework} as set up by the AutoML Benchmark',
        # todo: use something more thorough like for image names
        external_version=f'amlb=={__version__},{metadata.framework}=={metadata.framework_version}',
        # The values below are default values for a flow., the run will record used values.
        parameters=OrderedDict(
            time='240',
            memory='32',
            cores='8'
        ),
        parameters_meta_info=OrderedDict(
            time=dict(data_type='int', description='time in minutes'),
            memory=dict(data_type='int', description='memory in gigabytes'),
            cores=dict(data_type='int', description='number of available cores')
        ),
        language='English',
        # We can use components to describe subflows, e.g. the automl framework with its hyperparameters.
        # For now we don't.
        components=OrderedDict(),
        model=None,
        tags=["amlb"],
        dependencies=f'amlb=={__version__},{metadata.framework}=={metadata.framework_version}',
    )
    # If the flow does not yet exist on the server, it is registered.
    # Otherwise the local version is overwritten the
    return amlb_flow.publish()
#
#
# def _create_run(task_folder: str) -> openml.runs.OpenMLRun:
#     predictions = _load_predictions(task_folder)
#     metadata = _load_task_data(task_folder)
#
#  # prediction = format_prediction(
#  #        task=task,
#  #        repeat=repeat,
#  #        fold=fold,
#  #        index=index,
#  #        prediction=class_map[yp],
#  #        truth=y,
#  #        proba={c: pb for (c, pb) in zip(task.class_labels, proba)},
#  #    )
#
# def _upload_results(task_folder: str) -> openml.runs.OpenMLRun:
#     #
#     # oml_flow = _get_flow(metadata)
#     # oml_run = _create_run(metadata, predictions)
#     #
#     # # load meta-data
#     # # load predictions
#     # return openml.runs.OpenMLRun(
#     #     task_id=task_id, flow_id=flow_id, dataset_id=dataset_id,
#     #     parameter_settings=parameters,
#     #     setup_string=benchmark_command,
#     #     data_content=predictions,
#     #     tags=['study_218']
#     # )
#     pass
#
#
# def process_task_folder(task_folder: str) -> Optional[openml.runs.OpenMLRun]:
#     """ Uploads """
#     completed_folds = _list_completed_folds(task_folder)
#     is_ready_for_upload = (len(completed_folds) == 10)
#     if not is_ready_for_upload:
#         log.warning(
#             "Task %s is missing predictions for folds %s.",
#             task_folder,
#             ', '.join(completed_folds)
#         )
#         return None
#
#     return _upload_results(task_folder)
