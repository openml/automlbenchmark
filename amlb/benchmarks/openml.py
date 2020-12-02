import logging
from typing import List, Tuple, Optional

import openml

from amlb.utils import Namespace, str_sanitize


log = logging.getLogger(__name__)


def is_openml_benchmark(benchmark: str) -> bool:
    """ Check if 'benchmark' is a valid identifier for an openml task or suite. """
    if len(benchmark.split('/')) == 3:
        domain, oml_type, oml_id = benchmark.split('/')
        supported_types = ['s', 't']

        if oml_id.isdecimal():
            return domain == "openml" and oml_type in supported_types
    return False


def load_oml_benchmark(benchmark: str) -> Tuple[str, Optional[str], List[Namespace]]:
    """ Loads benchmark defined by openml suite or task, from openml/s/X or openml/t/Y. """
    domain, oml_type, oml_id = benchmark.split('/')
    path = None  # benchmark file does not exist on disk
    name = benchmark  # name is later passed as cli input again for containers, it needs to remain parsable
    if oml_type == 't':
        log.info("Loading openml task %s.", oml_id)
        # We first have the retrieve the task because we don't know the dataset id
        t = openml.tasks.get_task(oml_id, download_data=False)
        data = openml.datasets.get_dataset(t.dataset_id, download_data=False)
        tasks = [Namespace(name=str_sanitize(data.name),
                           description=data.description,
                           openml_task_id=t.id)]
    elif oml_type == 's':
        log.info("Loading openml suite %s.", oml_id)
        suite = openml.study.get_suite(oml_id)

        # Here we know the (task, dataset) pairs so only download dataset meta-data is sufficient
        tasks = []
        for tid, did in zip(suite.tasks, suite.data):
            data = openml.datasets.get_dataset(did, download_data=False)
            tasks.append(Namespace(name=str_sanitize(data.name),
                                   description=data.description,
                                   openml_task_id=tid))
    else:
        raise ValueError(f"The oml_type is {oml_type} but must be 's' or 't'")
    return name, path, tasks
