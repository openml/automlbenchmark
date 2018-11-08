import logging
import os
import time
import warnings

from autosklearn.classification import AutoSklearnClassifier
import autosklearn.metrics
from numpy import dtype
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

from automl.benchmark import TaskConfig
from automl.data import Dataset
from automl.results import save_predictions_to_file

log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    log.info("\n**** AutoSklearn ****\n")
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=DeprecationWarning)

    # Mapping of benchmark metrics to autosklearn metrics
    if config.metric == "acc":
        performance_metric = autosklearn.metrics.accuracy
    elif config.metric == "auc":
        performance_metric = autosklearn.metrics.roc_auc
    elif config.metric == "logloss":
        performance_metric = autosklearn.metrics.log_loss
    else:
        # TODO: Figure out if we are going to blindly pass metrics through, or if we use a strict mapping
        log.warning("Performance metric {} not supported.".format(config.metric))
        performance_metric = None

    # Set resources based on datasize
    log.warning("ignoring n_cores.")
    log.info("Running auto-sklearn with a maximum time of {}s on {} cores with {}MB, optimizing {}."
          .format(config.max_runtime_seconds, config.cores, config.max_mem_size_mb, performance_metric))

    X_train = dataset.train.X_enc
    y_train = dataset.train.y
    predictors_type = ['Categorical' if p.is_categorical() else 'Numerical' for p in dataset.predictors]

    log.warning("Using meta-learned initialization, which might be bad (leakage).")
    # TODO: Do we need to set per_run_time_limit too?
    start_time = time.time()
    auto_sklearn = AutoSklearnClassifier(time_left_for_this_task=config.max_runtime_seconds, ml_memory_limit=config.max_mem_size_mb)
    auto_sklearn.fit(X_train, y_train, metric=performance_metric, feat_type=predictors_type)
    actual_runtime_min = (time.time() - start_time)/60.0
    log.info("Requested training time (minutes): " + str((config.max_runtime_seconds/60.0)))
    log.info("Actual training time (minutes): " + str(actual_runtime_min))

    # Convert output to strings for classification
    log.info("Predicting on the test set.")
    X_true= dataset.test.X_enc
    y_true = dataset.test.y
    class_predictions = auto_sklearn.predict(X_true)
    class_probabilities = auto_sklearn.predict_proba(X_true)

    if class_predictions.dtype != dtype('<U32'):
        class_predictions = class_predictions.astype(int).astype(str)

    log.info("Optimization was towards metric, but following score is always accuracy.")
    log.info("Accuracy: " + str(accuracy_score(y_true, class_predictions)))

    if class_probabilities.shape[1] == 2:
        auc = roc_auc_score(y_true=y_true, y_score=class_probabilities[:,1])
        log.info("AUC: " + str(auc))
    else:
        logloss = log_loss(y_true=y_true, y_pred=class_probabilities)
        log.info("logloss: ", logloss)

    save_predictions_to_file(dataset=dataset,
                             output_file=config.output_file_template,
                             class_probabilities=class_probabilities,
                             class_predictions=class_predictions,
                             class_truth=y_true.values,
                             encode_classes=True)
