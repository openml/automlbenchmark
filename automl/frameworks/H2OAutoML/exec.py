import logging
import os
import time

from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.utils.multiclass import type_of_target

import h2o
from h2o.automl import H2OAutoML

from automl.benchmark import TaskConfig
from automl.data import Dataset
from automl.utils import encode_labels, save_predictions_to_file

log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    log.info("\n**** H2O AutoML ****\n")
    # Mapping of benchmark metrics to H2O metrics
    if config.metric == 'acc':
        h2o_metric = 'mean_per_class_error'
    elif config.metric == 'auc':
        h2o_metric = 'AUC'
    elif config.metric == 'logloss':
        h2o_metric = 'logloss'
    else:
        # TODO: Figure out if we are going to blindly pass metrics through, or if we use a strict mapping
        log.warning("Performance metric {} not supported, using AUTO.".format(config.metric))
        h2o_metric = None

    try:
        log.info("Starting H2O cluster.")
        log.debug("cores {}, memory {}mb".format(config.cores, config.max_mem_size_mb))
        h2o.init(nthreads=config.cores, max_mem_size=str(config.max_mem_size_mb) + "M")

        # Load train as an H2O Frame, but test as a Pandas DataFrame
        log.debug("Loading train data from {}".format(dataset.train.path))
        train = h2o.import_file(dataset.train.path)
        log.debug("Loading test data from {}".format(dataset.test.path))
        test = h2o.import_file(dataset.test.path)

        log.info("Running model on task {}, fold {}".format(config.name, config.fold))
        log.debug("Running H2O AutoML with a maximum time of {}s on {} core(s), optimizing {}."
              .format(config.max_runtime_seconds, config.cores, h2o_metric))
        start_time = time.time()

        aml = H2OAutoML(max_runtime_secs=config.max_runtime_seconds, sort_metric=h2o_metric)
        aml.train(y=dataset.target.index, training_frame=train)
        actual_runtime_min = (time.time() - start_time)/60.0
        log.debug("Requested training time (minutes): " + str((config.max_runtime_seconds/60.0)))
        log.info("Actual training time (minutes): " + str(actual_runtime_min))

        log.info("Predicting the test set.")
        predictions = aml.predict(test).as_data_frame()

        preview_size = 20
        # truth_df = test[:, -1].as_data_frame(header=False)
        truth_df = test[:, dataset.target.index].as_data_frame(header=False)
        predictions.insert(0, 'truth', truth_df)
        log.info("Predictions sample:\n %s\n", predictions.head(preview_size).to_string())

        y_pred = predictions.iloc[:, 1]
        y_true = predictions.iloc[:, 0]
        log.debug("test target type: "+type_of_target(y_true))
        accuracy = accuracy_score(y_true, y_pred)
        log.info("Optimization was towards metric, but following score is always accuracy.")
        log.info("Accuracy: "+str(accuracy))

        # TO DO: See if we can use the h2o-sklearn wrappers here instead
        class_predictions = y_pred.values
        class_probabilities = predictions.iloc[:, 2:].values

        # TO DO: Change this to roc_curve, auc
        if type(aml.leader.model_performance()) == h2o.model.metrics_base.H2OBinomialModelMetrics:
            y_true_binary, _ = encode_labels(y_true, dataset.target.values)
            y_scores = predictions.iloc[:, -1].values
            auc = roc_auc_score(y_true=y_true_binary, y_score=y_scores)
            log.info("AUC: " + str(auc))
        elif type(aml.leader.model_performance()) == h2o.model.metrics_base.H2OMultinomialModelMetrics:
            logloss = log_loss(y_true=y_true, y_pred=class_probabilities)
            log.info("Log Loss: " + str(logloss))

        dest_file = os.path.join(os.path.expanduser(config.output_dir), "predictions_h2o_{task}_{fold}.txt".format(task=config.name, fold=config.fold))
        save_predictions_to_file(class_probabilities, class_predictions.astype(str), dest_file)
        log.info("Predictions saved to %s", dest_file)

    finally:
        if h2o.connection():
            h2o.connection().close()

