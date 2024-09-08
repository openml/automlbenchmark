from __future__ import annotations

import logging
from typing import cast

import openml
import pandas as pd

from amlb.utils import Namespace, str_sanitize


log = logging.getLogger(__name__)


def is_openml_benchmark(benchmark: str) -> bool:
    """ Check if 'benchmark' is a valid identifier for an openml task or suite. """
    if len(benchmark.split('/')) == 3:
        domain, oml_type, oml_id = benchmark.split('/')
        supported_types = ['s', 't']

        if oml_id.isdecimal():
            return domain in ("openml", "test.openml") and oml_type in supported_types
    return False


def load_oml_benchmark(benchmark: str) -> tuple[str, str | None, list[Namespace]]:
    """ Loads benchmark defined by openml suite or task, from openml/s/X or openml/t/Y. """
    domain, oml_type, oml_id = benchmark.split('/')

    if domain == "test.openml":
        log.debug("Setting openml server to the test server.")
        openml.config.server = "https://test.openml.org/api/v1/xml"

    if openml.config.retry_policy != "robot":
        log.debug(
            "Setting openml retry_policy from '%s' to 'robot'." % openml.config.retry_policy)
        openml.config.set_retry_policy("robot")

    if oml_type == 't':
        log.info("Loading openml task %s.", oml_id)
        # We first have the retrieve the task because we don't know the dataset id
        t = openml.tasks.get_task(oml_id, download_data=False, download_qualities=False)
        data = openml.datasets.get_dataset(t.dataset_id, download_data=False, download_qualities=False)
        tasks = [Namespace(name=str_sanitize(data.name),
                           description=data.description,
                           openml_task_id=t.id,
                           id="{}.org/t/{}".format(domain, t.id))]
    elif oml_type == 's':
        log.info("Loading openml suite %s.", oml_id)
        suite = openml.study.get_suite(oml_id)

        # Here we know the (task, dataset) pairs so only download dataset meta-data is sufficient
        tasks = []
        datasets = cast(pd.DataFrame, openml.datasets.list_datasets(data_id=suite.data, output_format='dataframe'))
        datasets.set_index('did', inplace=True)
        for tid, did in zip(cast(list[int], suite.tasks), cast(list[int], suite.data)):
            tasks.append(Namespace(name=str_sanitize(datasets.loc[did]['name']),
                                   description=f"{openml.config.server.replace('/api/v1/xml', '')}/d/{did}",
                                   openml_task_id=tid,
                                   id="{}.org/t/{}".format(domain, tid)))
    else:
        raise ValueError(f"The oml_type is {oml_type} but must be 's' or 't'")
    # The first argument needs to remain parsable further in the pipeline as is
    # The second argument is path, the benchmark does not exist on disk
    return benchmark, None, tasks
