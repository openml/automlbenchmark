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
    log.info("\n**** ML-Plan for scikit-learn ****\n")

    is_classification = config.type == "classification"
    if not is_classification:
	    raise ValueError('ML-Plan for scikit-learn does not support regression')

    # Mapping of benchmark metrics to Weka metrics
    metrics_mapping = dict(
        acc='ERRORRATE',
        auc='AUC',
        logloss='LOGLOSS',
	    f1='F1'
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

    cmd_root = "java -jar {here}/lib/mlplan/mlplan-cli*.jar ".format(here=dir_of(__file__))
    cmd_params = dict(
	    m='sklearn',
        f='"{}"'.format(train_file),
        p='"{}"'.format(test_file),
        t=config.max_runtime_seconds,
        ncpus=config.cores,
        l=metric,
	    s=config.seed,   # weka accepts only int16 as seeds
	    ooab=config.output_predictions_file,
        **training_params
    )

    cmd = cmd_root + ' '.join(["-{} {}".format(k, v) for k, v in cmd_params.items()])

    with Timer() as training:
        run_cmd(cmd, _live_output_=True)

    return dict(
        training_duration=training.duration
    )

