import logging
from typing import List, Union, Tuple

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


def load_oml_benchmark(benchmark: str) -> Tuple[str, List[Namespace]]:
    domain, oml_type, oml_id = benchmark.split('/')
    if oml_type == 't':
        log.info("Loading openml task %s.", oml_id)
        # We first have the retrieve the task because we don't know the dataset id
        t = openml.tasks.get_task(oml_id, download_data=False)
        data = openml.datasets.get_dataset(t.dataset_id, download_data=False)
        return "task_{}".format(oml_id), [Namespace(name=data.name, description=data.description, openml_task_id=t.id)]
    elif oml_type == 's':
        log.info("Loading openml study %s.", oml_id)
        study = openml.study.get_suite(oml_id)
        name = "study_{}".format(oml_id) if study.alias is None else study.alias

        # Here we know the (task, dataset) pairs so only download dataset meta-data is sufficient
        tasks = []
        for tid, did in zip(study.tasks, study.data):
            data = openml.datasets.get_dataset(did, download_data=False)
            tasks.append(Namespace(name=data.name, description=data.description, openml_task_id=tid))

        return name, tasks
    else:
        raise ValueError("The oml_type is {} but must be 's' or 't'".format(oml_type))
