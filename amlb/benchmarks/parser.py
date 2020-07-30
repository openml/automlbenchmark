import os
from typing import Tuple, List

from amlb.utils import Namespace
from .openml import is_openml_benchmark, oml_benchmark_to_tasks
from .file import file_benchmark_to_tasks


def benchmark_load(name, benchmark_definition_dirs: List[str]):
    # Identify where the resource is located, all name structures are clearly defined,
    # but local file benchmark can require probing from disk to see if it is valid,
    # which is why it is tried last.
    if is_openml_benchmark(name):
        tasks = oml_benchmark_to_tasks(name)
        benchmark_name = "study_{}".format(name.split('/')[-1])
    # elif is_kaggle_benchmark(name):
    else:
        tasks = file_benchmark_to_tasks(name, benchmark_definition_dirs)
        benchmark_name, _ = os.path.splitext(os.path.basename(name))

    hard_defaults = next((task for task in tasks if task.name == '__defaults__'), None)
    tasks = [task for task in tasks if task is not hard_defaults]
    return hard_defaults, tasks, 'test_file_name'



