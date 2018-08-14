import sys

from hpsklearn import HyperoptEstimator, any_classifier
from hyperopt import tpe

sys.path.append('../common')
import common_code

if __name__ == '__main__':
    runtime_seconds = sys.argv[1]
    number_cores = int(sys.argv[2])
    performance_metric = sys.argv[3]

    X_train, y_train = common_code.get_X_y_from_arff(common_code.TRAIN_DATA_PATH)
    X_test, y_test = common_code.get_X_y_from_arff(common_code.TEST_DATA_PATH)
    X_train, X_test = X_train.astype(float), X_test.astype(float)

    print('Running hyperopt-sklearn with a maximum time of {}s on {} cores, optimizing {}.'
          .format(runtime_seconds, number_cores, performance_metric))

    print('ignoring runtime.')  # Not available? just number of iterations.
    print('ignoring n_cores.')  # Not available
    print('always optimize towards accuracy.')  # loss_fn lambda y1,y2:loss(y1, y2)
    hyperoptsklearn = HyperoptEstimator(classifier=any_classifier('clf'), algo=tpe.suggest)
    hyperoptsklearn.fit(X_train, y_train)
    class_predictions = hyperoptsklearn.predict(X_test)
    class_probabilities = common_code.one_hot_encode_predictions(class_predictions)

    common_code.save_predictions_to_file(class_probabilities, class_predictions)
