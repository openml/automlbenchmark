import sys

from autosklearn.classification import AutoSklearnClassifier
import autosklearn.metrics

sys.path.append('../common')
import common_code

if __name__ == '__main__':
    runtime_seconds = int(sys.argv[1])
    number_cores = int(sys.argv[2])
    performance_metric = sys.argv[3]

    X_train, y_train = common_code.get_X_y_from_arff(common_code.TRAIN_DATA_PATH)
    X_test, y_test = common_code.get_X_y_from_arff(common_code.TEST_DATA_PATH)
    X_train, X_test = X_train.astype(float), X_test.astype(float)

    print('Running auto-sklearn with a maximum time of {}s on {} cores, optimizing {}.'
          .format(runtime_seconds, number_cores, performance_metric))

    print('ignoring n_cores.')
    print('Using meta-learned initialization, which might be bad (leakage).')
    auto_sklearn = AutoSklearnClassifier(time_left_for_this_task=runtime_seconds)
    print('always optimize towards accuracy.')
    auto_sklearn.fit(X_train, y_train, metric=autosklearn.metrics.accuracy)
    class_predictions = auto_sklearn.predict(X_test)
    class_probabilities = auto_sklearn.predict_proba(X_test)

    common_code.save_predictions_to_file(class_probabilities, class_predictions)
