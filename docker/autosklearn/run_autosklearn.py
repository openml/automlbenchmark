from autosklearn.classification import AutoSklearnClassifier
import autosklearn.metrics
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import time
import warnings

import sys
sys.path.append('/bench/common')
import common_code


if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=DeprecationWarning)

    runtime_seconds = int(sys.argv[1])
    number_cores = int(sys.argv[2])
    performance_metric = sys.argv[3]
    memory_limit_mb = int(sys.argv[4])

    # Mapping of benchmark metrics to autosklearn metrics
    if performance_metric == "acc":
        performance_metric = autosklearn.metrics.accuracy
    elif performance_metric == "auc":
        performance_metric = autosklearn.metrics.roc_auc
    elif performance_metric == "logloss":
        performance_metric = autosklearn.metrics.log_loss   
    else:
        # TO DO: Figure out if we are going to blindly pass metrics through, or if we use a strict mapping
        print('Performance metric, {}, not supported.'.format(performance_metric))

    X_train, y_train = common_code.get_X_y_from_arff(common_code.TRAIN_DATA_PATH)
    X_test, y_test = common_code.get_X_y_from_arff(common_code.TEST_DATA_PATH)
    X_train, X_test = X_train.astype(float), X_test.astype(float)

    # Set resources based on datasize
    print('ignoring n_cores.')
    print('Running auto-sklearn with a maximum time of {}s on {} cores with {}MB, optimizing {}.'
          .format(runtime_seconds, number_cores, memory_limit_mb, performance_metric))

    print('Using meta-learned initialization, which might be bad (leakage).')
    # TO DO: Do we need to set per_run_time_limit too?
    starttime = time.time()
    auto_sklearn = AutoSklearnClassifier(time_left_for_this_task=runtime_seconds, \
        ml_memory_limit=memory_limit_mb)
    auto_sklearn.fit(X_train, y_train, metric=performance_metric)
    actual_runtime_min = (time.time() - starttime)/60.0
    print('Requested training time (minutes): ' + str((runtime_seconds/60.0)))
    print('Actual training time (minutes): ' + str(actual_runtime_min))    


    # Convert output to strings for classification
    print('Predicting on the test set.')
    class_predictions = auto_sklearn.predict(X_test)
    class_probabilities = auto_sklearn.predict_proba(X_test)

    print('Optimization was towards metric, but following score is always accuracy:')
    print("Accuracy: " + str(accuracy_score(y_test, class_predictions)))  

    if class_probabilities.shape[1] == 2:
        auc = roc_auc_score(y_true=y_test.astype(int), y_score=class_probabilities[:,1])
        print("AUC: " + str(auc))
    else:
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder().fit(y_train)
        y_test = label_encoder.transform(y_test).reshape(-1, 1)
        logloss = log_loss(y_true=y_test, y_pred=class_probabilities)
        print('logloss: ', logloss)

    common_code.save_predictions_to_file(class_probabilities, class_predictions)
