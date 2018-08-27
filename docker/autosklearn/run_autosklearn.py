import sys

from autosklearn.classification import AutoSklearnClassifier
import autosklearn.metrics
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


sys.path.append('/bench/common')
import common_code

if __name__ == '__main__':

    runtime_seconds = int(sys.argv[1])
    number_cores = int(sys.argv[2])
    performance_metric = sys.argv[3]

    # Mapping of benchmark metrics to autosklearn metrics
    if performance_metric == "acc":
        performance_metric = autosklearn.metrics.accuracy
    elif performance_metric == "auc":
        performance_metric = autosklearn.metrics.roc_auc
    else:
        # TO DO: Figure out if we are going to blindly pass metrics through, or if we use a strict mapping
        print('Performance metric, {}, not supported.'.format(performance_metric))     

    X_train, y_train = common_code.get_X_y_from_arff(common_code.TRAIN_DATA_PATH)
    X_test, y_test = common_code.get_X_y_from_arff(common_code.TEST_DATA_PATH)
    X_train, X_test = X_train.astype(float), X_test.astype(float)

    # Set resources based on datasize
    print('ignoring n_cores.')
    # If small data:
    if len(y_train) <= 20000:
        number_cores = 8
        ml_memory_limit = 16000 #16GB
    elif len(y_train) <= 200000:
        number_cores = 32
        ml_memory_limit = 64000 #64GB
    else:
        number_cores = 64
        ml_memory_limit = 640000 #64GB

    print('Running auto-sklearn with a maximum time of {}s on {} cores, optimizing {}.'
          .format(runtime_seconds, number_cores, performance_metric))

    print('Using meta-learned initialization, which might be bad (leakage).')
    auto_sklearn = AutoSklearnClassifier(time_left_for_this_task=runtime_seconds, \
        per_run_time_limit=runtime_seconds, \
        ml_memory_limit=ml_memory_limit)
    print('always optimize towards accuracy.')
    auto_sklearn.fit(X_train, y_train, metric=performance_metric)


    # Convert output to strings for classification
    print('Predicting on the test set.')
    class_predictions = auto_sklearn.predict(X_test).astype(np.int_).astype(np.str_)
    class_probabilities = auto_sklearn.predict_proba(X_test)

    print('Optimization was towards metric, but following score is always accuracy:')
    print("Accuracy: " + str(accuracy_score(y_test, class_predictions)))  

    if class_probabilities.shape[1] == 2:
        auc = roc_auc_score(y_true=y_test.astype(np.int_), y_score=class_probabilities[:,1])
        print("AUC: " + str(auc))

    common_code.save_predictions_to_file(class_probabilities, class_predictions)
