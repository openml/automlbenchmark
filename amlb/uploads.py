import json
import logging
import os
from collections import OrderedDict
from typing import Set, Optional, List

import openml
import pandas as pd
from openml import OpenMLTask, OpenMLFlow
from openml.runs.functions import format_prediction

from .utils.core import Namespace
from .__version__ import __version__


log = logging.getLogger(__name__)


def _load_task_data(task_folder: str, fold: int = 0) -> Namespace:
    """ Loads the metadata of the given fold of a task as a namespace. """
    with open(os.path.join(task_folder, str(fold), 'metadata.json'), 'r') as fh:
        metadata = json.load(fh)
    metadata = Namespace.from_dict(metadata)
    return metadata


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


def _list_completed_folds(task_folder: str) -> Set[str]:
    completed_folds = set()
    for fold_dir in os.listdir(task_folder):
        if "predictions.csv" in os.listdir(os.path.join(task_folder, fold_dir)):
            completed_folds.add(fold_dir)
    return completed_folds


def _get_flow(metadata: Namespace, sync_with_server: bool = True) -> openml.flows.OpenMLFlow:
    """ Creates or retrieves an OpenML flow for the given run metadata. """
    amlb_flow = openml.flows.OpenMLFlow(
        name=f"amlb_{metadata.framework}",
        description=f'{metadata.framework} as set up by the AutoML Benchmark',
        # todo: use something more thorough like for image names
        external_version=f'amlb=={__version__},{metadata.framework}=={metadata.framework_version}',
        # The values below are default values for a flow., the run will record used values.
        parameters=OrderedDict(
            max_runtime_seconds='14400',
            max_mem_size_mb='32768',
            cores='8',
            seed='42',
        ),
        parameters_meta_info=OrderedDict(
            max_runtime_seconds=dict(data_type='int', description='Maximum runtime in seconds.'),
            max_mem_size_mb=dict(data_type='int', description='Memory constraint in megabytes.'),
            cores=dict(data_type='int', description='Number of available cores.'),
            seed=dict(data_type='int', description='The random seed.')
        ),
        language='English',
        # We can use components to describe subflows, e.g. the automl framework with its hyperparameters.
        # For now we don't.
        components=OrderedDict(),
        model=None,
        tags=["amlb"],
        dependencies=f'amlb=={__version__},{metadata.framework}=={metadata.framework_version}',
    )
    if sync_with_server:
        # Publish will check if the flow exists on the server.
        # If it exists, the local object is updated with server information.
        # Otherwise the new flow is stored on the server.
        return amlb_flow.publish()
    else:
        return amlb_flow


def _extract_and_format_hyperparameter_configuration(metadata: Namespace, flow: OpenMLFlow) -> List[OrderedDict]:
    return [
        OrderedDict([
            ("oml:name", name),
            ("oml:value", metadata.__dict__.get(name, default)),
            ("oml:component", flow.id)
        ])
        for name, default in flow.parameters.items()
    ]


def _upload_results(task_folder: str) -> openml.runs.OpenMLRun:
    metadata = _load_task_data(task_folder)
    predictions = _load_predictions(task_folder)

    oml_flow = _get_flow(metadata)
    oml_task = openml.tasks.get_task(metadata.openml_task_id)

    denormalize_map = {label.strip().lower(): label for label in oml_task.class_labels}
    predictions.columns = [col if col not in denormalize_map else denormalize_map[col] for col in predictions]

    formatted_predictions = []
    for _, row in predictions.iterrows():
        if metadata.type != "classification":
            class_probabilities = None
        else:
            class_probabilities = {c: row[c] for c in oml_task.class_labels}
        prediction = format_prediction(
               task=oml_task,
               repeat=row["repeat"],
               fold=row["fold"],
               index=row["index"],
               prediction=row["predictions"],
               truth=row["truth"],
               proba=class_probabilities,
           )
        formatted_predictions.append(prediction)

    parameters = _extract_and_format_hyperparameter_configuration(metadata, oml_flow)
    tags = ['amlb']
    if 'tag' in metadata and metadata.tag not in [None, 'amlb']:
        tags.extend([metadata.tag])

    return openml.runs.OpenMLRun(
        task_id=oml_task.id,
        flow_id=oml_flow.id,
        dataset_id=oml_task.dataset_id,
        parameter_settings=parameters,
        setup_string=metadata.command,
        data_content=formatted_predictions,
        tags=tags,
    ).publish()


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
