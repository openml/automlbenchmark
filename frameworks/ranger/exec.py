import logging
import os

from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.datautils import read_csv
from amlb.utils import dir_of, run_cmd

log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    #TODO: use rpy2 instead? not necessary here though as the call is very simple
    log.info("\n**** Random Forest (R) ****\n")

    here = dir_of(__file__)
    meta_results_file = os.path.join(os.path.dirname(config.output_metadata_file), "meta_results.csv")
    run_cmd(("Rscript --vanilla -e \""
             ".libPaths('{package_directory}'); source('{script}'); "
             "run('{train}', '{test}', '{output}', cores={cores}, meta_results_file='{meta_results}', task_type='{task_type}')"
             "\"").format(
        package_directory=os.path.join(here, "lib"),
        script=os.path.join(here, 'exec.R'),
        train=dataset.train.path,
        test=dataset.test.path,
        output=config.output_predictions_file,
        meta_results=meta_results_file,
        task_type=config.type,
        cores=config.cores
    ), _live_output_=True)

    log.info("Predictions saved to %s", config.output_predictions_file)

    meta_results = read_csv(meta_results_file)
    return dict(
        training_duration=meta_result(meta_results, 'training_duration'),
        predict_duration=meta_result(meta_results, 'predict_duration')
    )


def meta_result(df, key):
    return df.loc[df['key'] == key, 'value'].squeeze()

