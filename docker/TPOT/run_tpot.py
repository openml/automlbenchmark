import sys

from tpot import TPOTClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.append('/bench/common')
import common_code

if __name__ == '__main__':

    runtime_seconds = sys.argv[1]
    number_cores = int(sys.argv[2])
    performance_metric = sys.argv[3]

     # Mapping of benchmark metrics to TPOT metrics
    if performance_metric == "acc":
        performance_metric = "accuracy"
    elif performance_metric == "auc":
        performance_metric = "roc_auc"
    else:
        # TO DO: Figure out if we are going to blindly pass metrics through, or if we use a strict mapping
        print('Performance metric, {}, not supported.'.format(performance_metric))

    X_train, y_train = common_code.get_X_y_from_arff(common_code.TRAIN_DATA_PATH)
    X_test, y_test = common_code.get_X_y_from_arff(common_code.TEST_DATA_PATH)
    X_train, X_test = X_train.astype(float), X_test.astype(float)

    # Convert response from string to integers
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    performance_metric = 'accuracy' if performance_metric=='acc' else performance_metric

    print('Running TPOT with a maximum time of {}s on {} cores, optimizing {}.'
          .format(runtime_seconds, number_cores, performance_metric))

    runtime_min = (int(runtime_seconds)/60)
    tpot = TPOTClassifier(n_jobs=number_cores, \
                          max_time_mins=runtime_min, \
                          verbosity=2, \
                          scoring=performance_metric)
    tpot.fit(X_train, y_train)

    print('Predicting on the test set.')
    class_predictions = tpot.predict(X_test)
    try:
        class_probabilities = tpot.predict_proba(X_test)
    except RuntimeError:
        # TPOT throws a RuntimeError if the optimized pipeline does not support `predict_proba`.
        class_probabilities = common_code.one_hot_encode_predictions(class_predictions)

    print('Optimization was towards metric, but following score is always accuracy:')
    print("Accuracy: " + str(accuracy_score(y_test, class_predictions)))

    if class_probabilities.shape[1] == 2:
        auc = roc_auc_score(y_true=y_test.astype(int), y_score=class_probabilities[:,1])
        print("AUC: " + str(auc))

    common_code.save_predictions_to_file(class_probabilities, class_predictions)
