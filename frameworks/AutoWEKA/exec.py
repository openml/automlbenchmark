import logging

from automl.benchmark import TaskConfig
from automl.data import Dataset
from automl.datautils import reorder_dataset
from automl.results import save_predictions_to_file
from automl.utils import dir_of, path_from_split, run_cmd, split_path

log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    log.info("\n**** AutoWEKA ****\n")

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
    output = run_cmd("java -cp {here}/libs/autoweka/autoweka.jar weka.classifiers.meta.AutoWEKAClassifier -t {train} -T {test} -memLimit {max_memory} \
    -classifications \"weka.classifiers.evaluation.output.prediction.CSV -distribution -file {predictions_output}\" \
    -timeLimit {time} -parallelRuns {cores} -metric {metric}".format(
        here=dir_of(__file__),
        train=train_file,
        test=test_file,
        max_memory=config.max_mem_size_mb,
        time=int(config.max_runtime_seconds/60),
        cores=config.cores,
        metric=metric,
        predictions_output=weka_file
    ))
    log.debug(output)

    # if target values are not sorted alphabetically in the ARFF file, then class probabilities are returned in the original order
    # interestingly, other frameworks seem to always sort the target values first
    # that's why we need to specify the probabilities labels here: sorting+formatting is done in saving function
    class_probabilities_labels = dataset.target.values
    with open(weka_file, 'r') as weka_file:
        class_probabilities = []
        class_predictions = []
        class_truth = []
        for line in weka_file.readlines()[1:-1]:
            inst, actual, predicted, error, *distribution = line.split(',')
            pred_probabilities = [pred_probability.replace('*', '').replace('\n', '') for pred_probability in distribution]
            _, pred_class = predicted.split(':')
            _, truth = actual.split(':')
            class_probabilities.append(pred_probabilities)
            class_predictions.append(pred_class)
            class_truth.append(truth)

    save_predictions_to_file(dataset=dataset,
                             output_file=config.output_predictions_file,
                             class_probabilities=class_probabilities,
                             class_predictions=class_predictions,
                             class_truth=class_truth,
                             class_probabilities_labels=class_probabilities_labels)
