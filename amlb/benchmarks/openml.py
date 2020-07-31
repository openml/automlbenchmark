import logging
from typing import List, Tuple, Optional

import openml

from amlb.utils import Namespace


log = logging.getLogger(__name__)


def is_openml_benchmark(benchmark: str) -> bool:
    """ Check if 'benchmark' is a valid identifier for an openml task or study. """
    if len(benchmark.split('/')) == 3:
        domain, oml_type, oml_id = benchmark.split('/')
        supported_types = ['s', 't']

        valid_id = False
        try:
            _ = int(oml_id)
            valid_id = True
        except ValueError:
            pass

        return domain == "openml" or oml_type in supported_types and valid_id
    return False


def load_oml_benchmark(benchmark: str) -> Tuple[str, Optional[str], List[Namespace]]:
    """ Loads benchmark defined by openml study or task, from openml/s/X or openml/t/Y. """
    domain, oml_type, oml_id = benchmark.split('/')
    path = None  # benchmark file does not exist on disk
    name = benchmark  # name is later passed as cli input again for containers, it needs to remain parsable
    if oml_type == 't':
        log.info("Loading openml task %s.", oml_id)
        # We first have the retrieve the task because we don't know the dataset id
        t = openml.tasks.get_task(oml_id, download_data=False)
        data = openml.datasets.get_dataset(t.dataset_id, download_data=False)
        tasks = [Namespace(name=data.name, description=data.description, openml_task_id=t.id)]
    elif oml_type == 's':
        log.info("Loading openml study %s.", oml_id)
        study = openml.study.get_suite(oml_id)

        # Here we know the (task, dataset) pairs so only download dataset meta-data is sufficient
        tasks = []
        for tid, did in zip(study.tasks, study.data):
            data = openml.datasets.get_dataset(did, download_data=False)
            tasks.append(Namespace(name=data.name, description=data.description, openml_task_id=tid))
    else:
        raise ValueError("The oml_type is {} but must be 's' or 't'".format(oml_type))
    return name, path, tasks
