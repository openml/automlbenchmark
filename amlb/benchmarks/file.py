import logging
import os
from typing import List

from amlb.utils import config_load


log = logging.getLogger(__name__)


def _find_local_benchmark_definition(name: str, benchmark_definition_dirs: List[str]) -> str:
    # 'name' should be either a full path to the benchmark,
    # or a filename (without extension) in the benchmark directory.
    if os.path.exists(name):
        return name

    for bd in benchmark_definition_dirs:
        bf = os.path.join(bd, "{}.yaml".format(name))
        if os.path.exists(bf):
            # We don't account for duplicate definitions (yet).
            return bf

    # should we support s3 and check for s3 path before raising error?
    raise ValueError(
        "Incorrect benchmark name or path `{}`, name not available in {}.".format(
            name, benchmark_definition_dirs))


def load_file_benchmark(name: str, benchmark_definition_dirs: List[str]):
    benchmark_file = _find_local_benchmark_definition(name, benchmark_definition_dirs)
    log.info("Loading benchmark definitions from %s.", benchmark_file)
    tasks = config_load(benchmark_file)
    benchmark_name, _ = os.path.splitext(os.path.basename(benchmark_file))
    return benchmark_name, tasks