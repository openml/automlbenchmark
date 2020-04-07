import logging

from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from frameworks.autogluon import exec_template

log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    STACKING = {
        'auto_stack': True,
    }
    parameters = STACKING
    return exec_template.run(dataset=dataset, config=config, parameters=parameters)
