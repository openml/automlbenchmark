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


def _load_predictions(task: OpenMLTask, prediction_files: List[str]) -> pd.DataFrame:
    results = []
    for prediction_file in prediction_files:
        results.append(pd.read_csv(prediction_file, sep=',', header=0))
    return pd.concat(results)


def _load_task_data(task_folder: str) -> Tuple[Namespace, pd.DataFrame]:
    """ Loads the metadata as a namespace, and predictions as dataframe. """
    with open(os.path.join(task_folder, '0', 'metadata.json'), 'r') as fh:
        metadata = json.load(fh)
    metadata = Namespace.from_dict(metadata)
    prediction_files = []
    predictions = _load_predictions(None, prediction_files)
    return task, metadata, predictions


def _list_completed_folds(task_folder: str) -> Set[str]:
    completed_folds = set()
    for fold_dir in os.listdir(task_folder):
        if "predictions.csv" in os.listdir(os.path.join(task_folder, fold_dir)):
            completed_folds.add(fold_dir)
    return completed_folds


def _get_flow(metadata: Namespace) -> openml.flows.OpenMLFlow:
    amlb_flow = openml.flows.OpenMLFlow(
        name=f"amlb_{metadata.framework}",
        description=f'{metadata.framework} as set up by the AutoML Benchmark',
        # todo: use something more thorough like for image names
        external_version=f'amlb=={__version__},framework=={metadata.framework_version}',
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
        # components=OrderedDict(automl_tool=auto_sklearn_flow),
    )
    # If the flow does not yet exist on the server, it is registered.
    # Otherwise the local version is overwritten the
    return amlb_flow.publish()


def _create_run(metadata, predictions) -> openml.runs.OpenMLRun:
    pass





def _upload_results(task_folder: str) -> openml.runs.OpenMLRun:
    #
    # oml_flow = _get_flow(metadata)
    # oml_run = _create_run(metadata, predictions)
    #
    # # load meta-data
    # # load predictions
    # return openml.runs.OpenMLRun(
    #     task_id=task_id, flow_id=flow_id, dataset_id=dataset_id,
    #     parameter_settings=parameters,
    #     setup_string=benchmark_command,
    #     data_content=predictions,
    #     tags=['study_218']
    # )
    pass


def process_task_folder(task_folder: str) -> Optional[openml.runs.OpenMLRun]:
    """ Uploads """
    completed_folds = _list_completed_folds(task_folder)
    is_ready_for_upload = (len(completed_folds) == 10)
    if not is_ready_for_upload:
        log.warning(
            "Task %s is missing predictions for folds %s.",
            task_folder,
            ', '.join(completed_folds)
        )
        return None

    return _upload_results(task_folder)
