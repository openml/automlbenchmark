import sys

import arff
import numpy as np
from sklearn.metrics import accuracy_score
from tpot import TPOTClassifier

if __name__ == '__main__':
    train_data_path = "../common/train.arff"
    test_data_path = "../common/test.arff"

    runtime_seconds = sys.argv[1]
    number_cores = int(sys.argv[2])
    performance_metric = sys.argv[3]

    print('Loading data.')
    def get_X_y_from_arff(arff_file_path):
        with open(arff_file_path, 'r') as arff_data_file:
            data_arff = arff.load(arff_data_file)
            data = data_arff['data']
            data = np.asarray(data)
            return data[:, :-1], data[:, -1]

    X_train, y_train = get_X_y_from_arff(train_data_path)
    X_test, y_test = get_X_y_from_arff(test_data_path)
    X_train, X_test = X_train.astype(float), X_test.astype(float)

    print('Running TPOT with a maximum time of {}s on {} cores, optimizing {}.'
          .format(runtime_seconds, number_cores, performance_metric))

    print('Running model on task.')
    runtime_min = (int(runtime_seconds)/60)
    print('ignoring runtime.')
    tpot = TPOTClassifier(n_jobs=number_cores, population_size=10, generations=2, verbosity=2, scoring=performance_metric)
    tpot.fit(X_train, y_train)
    y_pred = tpot.predict(X_test)
    print('Optimization was towards metric, but following score is always accuracy:')
    print("THIS_IS_A_DUMMY_TOKEN " + str(accuracy_score(y_test, y_pred)))
