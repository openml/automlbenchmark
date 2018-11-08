import logging
import os

from automl.benchmark import TaskConfig
from automl.data import Dataset

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

    dest_file = config.output_file_template + '.pred'
    with open(weka_file, 'r') as weka_file, open(dest_file, 'w') as output_file:
        for line in weka_file.readlines()[1:-1]:
            inst, actual, predicted, error, *distribution = line.split(',')
            class_probabilities = [class_probability.replace('*', '').replace('\n', '') for class_probability in distribution]
            class_index, class_name = predicted.split(':')
            output_file.write(','.join(class_probabilities + [class_name + '\n']))

