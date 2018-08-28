import sys

from tpot import TPOTClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc

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

    print('Running TPOT with a maximum time of {}s on {} cores, optimizing {}.'
          .format(runtime_seconds, number_cores, performance_metric))

    runtime_min = (int(runtime_seconds)/60)
    tpot = TPOTClassifier(n_jobs=number_cores,
                          max_time_mins=runtime_min,
                          verbosity=2,
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
        class_names = common_code.get_class_names_from_arff(common_code.TRAIN_DATA_PATH)
        fpr, tpr, thresholds = roc_curve(y_test, class_probabilities[:, 1], pos_label=class_names[1])
        auc_score = auc(fpr, tpr)
        print("AUC: " + str(auc_score))

    common_code.save_predictions_to_file(class_probabilities, class_predictions)
