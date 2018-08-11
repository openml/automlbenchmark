import sys

import arff
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from tpot import TPOTClassifier

if __name__ == '__main__':
    train_data_path = "../common/train.arff"
    test_data_path = "../common/test.arff"
    output_path = "../common/predictions.csv"

    runtime_seconds = sys.argv[1]
    number_cores = int(sys.argv[2])
    performance_metric = sys.argv[3]

    performance_metric = 'accuracy' if performance_metric=='acc' else performance_metric
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
    class_predictions = tpot.predict(X_test)
    try:
        class_probabilities = tpot.predict_proba(X_test)
    except RuntimeError:
        # TPOT gives a RuntimeError in case the best pipeline can not predict probabilities.
        with open(train_data_path, 'r') as arff_data_file:
            arff_data = arff.load(arff_data_file)
            target_name, target_type = arff_data['attributes'][-1]

        le = LabelEncoder().fit(target_type)
        class_predictions_le = le.transform(class_predictions).reshape(-1, 1)
        class_probabilities = OneHotEncoder().fit_transform(class_predictions_le)
        class_probabilities = class_probabilities.todense()

    class_predictions = class_predictions.reshape(-1, 1)
    combined_predictions = np.hstack((class_probabilities, class_predictions)).astype(str)
    np.savetxt(output_path, combined_predictions, delimiter=',', fmt="%s")
