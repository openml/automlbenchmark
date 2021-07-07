import logging
import math
import os

from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.datautils import reorder_dataset
from amlb.results import NoResultError, save_predictions
from amlb.utils import dir_of, path_from_split, run_cmd, split_path, Timer

log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    log.info(f"\n**** AutoWEKA [v{config.framework_version}]****\n")

    is_classification = config.type == 'classification'
    if not is_classification:
        raise ValueError('Regression is not supported.')

    # Mapping of benchmark metrics to Weka metrics
    metrics_mapping = dict(
        acc='errorRate',
        auc='areaUnderROC',
        logloss='kBInformation'
    )
    metric = metrics_mapping[config.metric] if config.metric in metrics_mapping else None
    if metric is None:
        raise ValueError("Performance metric {} not supported.".format(config.metric))

    train_file = dataset.train.path
    test_file = dataset.test.path
    # Weka to requires target as the last attribute
    if dataset.target.index != len(dataset.predictors):
        train_file = reorder_dataset(dataset.train.path, target_src=dataset.target.index)
        test_file = reorder_dataset(dataset.test.path, target_src=dataset.target.index)

    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}
    parallelRuns = config.framework_params.get('_parallelRuns', config.cores)

    memLimit = config.framework_params.get('_memLimit', 'auto')
    if memLimit == 'auto':
        memLimit = max(min(config.max_mem_size_mb,
                           math.ceil(config.max_mem_size_mb / parallelRuns)),
                       1024)  # AutoWEKA default memLimit
    log.info("Using %sMB memory per run on %s parallel runs.", memLimit, parallelRuns)

    f = split_path(config.output_predictions_file)
    f.extension = '.weka_pred.csv'

    # WEKA is not included in later versions of Auto-WEKA, in that case it is downloaded
    # and extracted to the following location:
    weka_jar = f"{dir_of(__file__)}/lib/weka/weka.jar"

    weka_file = path_from_split(f)
    cmd_root = "java -cp {here}/lib/autoweka/autoweka.jar{weka_path} weka.classifiers.meta.AutoWEKAClassifier ".format(
        here=dir_of(__file__),
        weka_path=f":{weka_jar}" if os.path.isfile(weka_jar) else ""
    )
    cmd_params = dict(
        t='"{}"'.format(train_file),
        T='"{}"'.format(test_file),
        memLimit=memLimit,
        classifications='"weka.classifiers.evaluation.output.prediction.CSV -distribution -file \\\"{}\\\""'.format(weka_file),
        timeLimit=int(config.max_runtime_seconds/60),
        parallelRuns=parallelRuns,
        metric=metric,
        seed=config.seed % (1 << 16),   # weka accepts only int16 as seeds
        **training_params
    )
    cmd = cmd_root + ' '.join(["-{} {}".format(k, v) for k, v in cmd_params.items()])
    with Timer() as training:
        run_cmd(cmd, _live_output_=True)

    # if target values are not sorted alphabetically in the ARFF file, then class probabilities are returned in the original order
    # interestingly, other frameworks seem to always sort the target values first
    # that's why we need to specify the probabilities labels here: sorting+formatting is done in saving function
    probabilities_labels = dataset.target.values
    if not os.path.exists(weka_file):
        raise NoResultError("AutoWEKA failed producing any prediction.")
    with open(weka_file, 'r') as weka_file:
        probabilities = []
        predictions = []
        truth = []
        for line in weka_file.readlines()[1:-1]:
            inst, actual, predicted, error, *distribution = line.split(',')
            pred_probabilities = [pred_probability.replace('*', '').replace('\n', '') for pred_probability in distribution]
            _, pred = predicted.split(':')
            _, tru = actual.split(':')
            probabilities.append(pred_probabilities)
            predictions.append(pred)
            truth.append(tru)

    save_predictions(dataset=dataset,
                     output_file=config.output_predictions_file,
                     probabilities=probabilities,
                     predictions=predictions,
                     truth=truth,
                     probabilities_labels=probabilities_labels)

    return dict(
        training_duration=training.duration
    )

