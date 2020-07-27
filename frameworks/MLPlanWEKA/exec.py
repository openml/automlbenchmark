import logging
import math
import os

from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.datautils import reorder_dataset
from amlb.results import NoResultError, save_predictions_to_file
from amlb.utils import dir_of, path_from_split, run_cmd, split_path, Timer

log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    log.info("\n**** ML-Plan for WEKA ****\n")

    # Mapping of benchmark metrics to Weka metrics
    metrics_mapping = dict(
        acc='ERRORRATE',
        auc='AUC',
        logloss='LOGLOSS',
	f1='F1',
	rmse="ROOT_MEAN_SQUARED_ERROR",
	mse="MEAN_SQUARED_ERROR",
	mae="MEAN_ABSOLUTE_ERROR",
	rmsle="ROOT_MEAN_SQUARED_LOGARITHM_ERROR",
	r2="R2"
    )
    metric = metrics_mapping[config.metric] if config.metric in metrics_mapping else None
    if metric is None:
	    raise ValueError('Performance metric {} is not supported.'.format(config.metric))

    train_file = dataset.train.path
    test_file = dataset.test.path
    # Weka requires target as the last attribute
    if dataset.target.index != len(dataset.predictors):
        train_file = reorder_dataset(dataset.train.path, target_src=dataset.target.index)
        test_file = reorder_dataset(dataset.test.path, target_src=dataset.target.index)

    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}

    mem_limit = str(max(config.max_mem_size_mb-1024,2048))
    log.info("Using {}MB as maximum memory for java".format(mem_limit))

    cmd_root = "java -jar -Xmx"+ mem_limit +"M {here}/lib/mlplan/mlplan-cli*.jar ".format(here=dir_of(__file__))
    cmd_params = dict(
        f='"{}"'.format(train_file),
        p='"{}"'.format(test_file),
        t=config.max_runtime_seconds,
        ncpus=config.cores,
        l=metric,
	    s=config.seed,   # weka accepts only int16 as seeds
	    ooab=config.output_predictions_file,
        **training_params
    )
    if config.type == 'regression':
	    cmd_params.update({"m": "weka-regression"})

    cmd = cmd_root + ' '.join(["-{} {}".format(k, v) for k, v in cmd_params.items()])

    with Timer() as training:
        run_cmd(cmd, _live_output_=True)

    return dict(
        training_duration=training.duration
    )

