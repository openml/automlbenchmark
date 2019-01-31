import logging

from automl.benchmark import TaskConfig
from automl.data import Dataset
from automl.datautils import reorder_dataset
from automl.results import save_predictions_to_file
from automl.utils import dir_of, path_from_split, run_cmd, split_path

log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    log.info("\n**** AutoWEKA ****\n")

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

    f = split_path(config.output_predictions_file)
    f.extension = '.weka_pred.csv'
    weka_file = path_from_split(f)
    cmd_root = "java -cp {here}/libs/autoweka/autoweka.jar weka.classifiers.meta.AutoWEKAClassifier ".format(here=dir_of(__file__))
    cmd_params = dict(
        t=train_file,
        T=test_file,
        memLimit=config.max_mem_size_mb,
        classifications='"weka.classifiers.evaluation.output.prediction.CSV -distribution -file {}"'.format(weka_file),
        timeLimit=int(config.max_runtime_seconds/60),
        parallelRuns=config.cores,
        metric=metric,
        **config.framework_params
    )
    cmd = cmd_root + ' '.join(["-{} {}".format(k, v) for k, v in cmd_params.items()])
    output = run_cmd(cmd)
    log.debug(output)

    # if target values are not sorted alphabetically in the ARFF file, then class probabilities are returned in the original order
    # interestingly, other frameworks seem to always sort the target values first
    # that's why we need to specify the probabilities labels here: sorting+formatting is done in saving function
    probabilities_labels = dataset.target.values
    with open(weka_file, 'r') as weka_file:
        probabilities = []
        predictions = []
        truth = []
        for line in weka_file.readlines()[1:-1]:
            inst, actual, predicted, error, *distribution = line.split(',')
            pred_probabilities = [pred_probability.replace('*', '').replace('\n', '') for pred_probability in distribution]
            _, pred = predicted.split(':')
            _, truth = actual.split(':')
            probabilities.append(pred_probabilities)
            predictions.append(pred)
            truth.append(truth)

    save_predictions_to_file(dataset=dataset,
                             output_file=config.output_predictions_file,
                             probabilities=probabilities,
                             predictions=predictions,
                             truth=truth,
                             probabilities_labels=probabilities_labels)
