import logging
import os

from automl.benchmark import TaskConfig
from automl.data import Dataset
from automl.utils import dir_of

log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    #TODO use rpy2 instead? not necessary here though as the call is very simple
    log.info("\n**** Random Forest (R) ****\n")

    dest_file = os.path.join(os.path.expanduser(config.output_dir), "predictions_random_forest_r_{task}_{fold}.txt".format(task=config.name, fold=config.fold))
    here = dir_of(__file__)
    output = os.popen(r"""Rscript --vanilla -e "source('{script}'); run('{train}', '{test}', '{output}', {cores})" """.format(
        script=os.path.join(here, 'exec.R'),
        train=dataset.train.path,
        test=dataset.test.path,
        output=dest_file,
        cores=config.cores
    )).read()
    log.debug(output)

    #todo: accuracy

    log.info("Predictions saved to %s", dest_file)
