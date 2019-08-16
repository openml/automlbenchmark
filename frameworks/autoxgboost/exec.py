import logging
import os

from automl.benchmark import TaskConfig
from automl.data import Dataset
from automl.utils import dir_of, run_cmd

log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    #TODO: use rpy2 instead? not necessary here though as the call is very simple
    log.info("\n**** Autoxgboost (R) ****\n")

    is_classification = config.type == 'classification'
    if not is_classification:
        raise ValueError('Regression is not supported.')

    here = dir_of(__file__)
    run_cmd(r"""Rscript --vanilla -e "source('{script}'); run('{train}', '{test}', '{output}', {cores}, time.budget = {time_budget})" """.format(
      script = os.path.join(here, 'exec.R'),
      train = dataset.train.path,
      test = dataset.test.path,
      output = config.output_predictions_file,
      cores = config.cores,
      time_budget = config.max_runtime_seconds
    ))

    log.info("Predictions saved to %s", config.output_predictions_file)
