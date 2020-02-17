import logging

from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from frameworks.autogluon import exec_template

from autogluon_utils.benchmarking.baselines.autogluon.ablated_autogluon_config import NO_STACK

log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    if config.fold != 0:
        raise AssertionError('config.fold should only be 0 when running AutoGluon ablations! Value: %s' % config.fold)
    parameters = NO_STACK
    return exec_template.run(dataset=dataset, config=config, parameters=parameters)
