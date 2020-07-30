import logging
from typing import List, Union

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


def oml_benchmark_to_tasks(benchmark: str) -> List[Namespace]:
    domain, oml_type, oml_id = benchmark.split('/')
    if oml_type == 't':
        log.info("Loading openml task %s.", oml_id)
        return [oml_task_to_ns(oml_id)]
    elif oml_type == 's':
        log.info("Loading openml study %s.", oml_id)
        study = openml.study.get_suite(oml_id)
        return [oml_task_to_ns(task_id) for task_id in study.tasks]
    else:
        raise ValueError("The oml_type is {} but must be 's' or 't'".format(oml_type))


def oml_task_to_ns(task_id: Union[str, int]) -> Namespace:
    t = openml.tasks.get_task(task_id, download_data=False)
    data = openml.datasets.get_dataset(t.dataset_id, download_data=False)
    return Namespace(name=data.name, description=data.description, openml_task_id=t.id)
