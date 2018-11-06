import os
import time

from sklearn.metrics import accuracy_score, roc_curve, auc
from tpot import TPOTClassifier

from automl.benchmark import TaskConfig
from automl.data import Dataset
from automl.utils import one_hot_encode_predictions, save_predictions_to_file


def run(dataset: Dataset, config: TaskConfig):
    print("\n**** TPOT ****\n")

    # Mapping of benchmark metrics to TPOT metrics
    if config.metric == 'acc':
        metric = 'accuracy'
    elif config.metric == 'auc':
        metric = 'roc_auc'
    elif config.metric == 'logloss':
        metric = 'neg_log_loss'
    else:
        raise ValueError("Performance metric {} not supported.".format(config.metric))

    X_train = dataset.train.X_enc.astype(float)
    y_train = dataset.train.y_enc.astype(int)
    X_test = dataset.test.X_enc.astype(float)
    y_test = dataset.test.y_enc.astype(int)

    print('Running TPOT with a maximum time of {}s on {} cores, optimizing {}.'
          .format(config.max_runtime_seconds, config.cores, metric))

    runtime_min = (config.max_runtime_seconds/60)
    tpot = TPOTClassifier(n_jobs=config.cores,
                          max_time_mins=runtime_min,
                          verbosity=2,
                          scoring=metric)
    start_time = time.time()
    tpot.fit(X_train, y_train)
    actual_runtime_min = (time.time() - start_time)/60.0
    print('Requested training time (minutes): ' + str(runtime_min))
    print('Actual training time (minutes): ' + str(actual_runtime_min))

    print('Predicting on the test set.')
    class_predictions = tpot.predict(X_test)
    try:
        class_probabilities = tpot.predict_proba(X_test)
    except RuntimeError:
        # TPOT throws a RuntimeError if the optimized pipeline does not support `predict_proba`.
        class_probabilities = one_hot_encode_predictions(class_predictions, dataset.target)

    print('Optimization was towards metric, but following score is always accuracy:')
    print("Accuracy: " + str(accuracy_score(y_test, class_predictions)))

    if class_probabilities.shape[1] == 2:
        fpr, tpr, thresholds = roc_curve(dataset.test.y, class_probabilities[:, 1], pos_label=dataset.target.values[1])
        auc_score = auc(fpr, tpr)
        print("AUC: " + str(auc_score))

    class_predictions = dataset.target.decode(class_predictions)

    dest_file = os.path.join(os.path.expanduser(config.output_folder), "predictions_tpot_{task}_{fold}.txt".format(task=config.name, fold=config.fold))
    save_predictions_to_file(class_probabilities, class_predictions, dest_file)
    print("Predictions saved to "+dest_file)
    print()
