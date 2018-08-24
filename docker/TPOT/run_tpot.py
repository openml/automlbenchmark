import sys

from tpot import TPOTClassifier
from sklearn.metrics import accuracy_score

sys.path.append('/bench/common')
import common_code

if __name__ == '__main__':

    runtime_seconds = sys.argv[1]
    number_cores = int(sys.argv[2])
    performance_metric = sys.argv[3]

    X_train, y_train = common_code.get_X_y_from_arff(common_code.TRAIN_DATA_PATH)
    X_test, y_test = common_code.get_X_y_from_arff(common_code.TEST_DATA_PATH)
    X_train, X_test = X_train.astype(float), X_test.astype(float)

    performance_metric = 'accuracy' if performance_metric=='acc' else performance_metric

    print('Running TPOT with a maximum time of {}s on {} cores, optimizing {}.'
          .format(runtime_seconds, number_cores, performance_metric))

    runtime_min = (int(runtime_seconds)/60)
    tpot = TPOTClassifier(n_jobs=number_cores, max_time_mins=runtime_min, verbosity=2, scoring=performance_metric)
    tpot.fit(X_train, y_train)
    class_predictions = tpot.predict(X_test)
    try:
        class_probabilities = tpot.predict_proba(X_test)
    except RuntimeError:
        # TPOT throws a RuntimeError if the optimized pipeline does not support `predict_proba`.
        class_probabilities = common_code.one_hot_encode_predictions(class_predictions)

    print('Optimization was towards metric, but following score is always accuracy:')

    common_code.save_predictions_to_file(class_probabilities, class_predictions)
