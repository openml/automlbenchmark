import logging
import os

from automl.benchmark import TaskConfig
from automl.data import Dataset
from automl.results import save_predictions_to_file

log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    log.info("\n**** AutoWEKA ****\n")

    # Mapping of benchmark metrics to Weka metrics
    if config.metric == 'acc':
        metric = 'errorRate'
    elif config.metric == 'auc':
        metric = 'areaUnderROC'
    elif config.metric == 'logloss':
        metric = 'kBInformation'
    else:
        raise ValueError("Performance metric {} not supported.".format(config.metric))

    weka_file = config.output_file_template + '.weka_pred'
    output = os.popen("java -cp ./libs/autoweka/autoweka.jar weka.classifiers.meta.AutoWEKAClassifier -t {train} -T {test} -memLimit {max_memory} \
    -classifications \"weka.classifiers.evaluation.output.prediction.CSV -distribution -file {predictions_output}\" \
    -timeLimit {time} -parallelRuns {cores} -metric {metric}".format(
        train=dataset.train.path,
        test=dataset.test.path,
        max_memory=config.max_mem_size_mb,
        time=int(config.max_runtime_seconds/60),
        cores=config.cores,
        metric=metric,
        predictions_output=weka_file
    )).read()

    log.info(output)

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
                             output_file=config.output_file_template,
                             class_probabilities=class_probabilities,
                             class_predictions=class_predictions,
                             class_truth=class_truth,
                             encode_classes=True)
