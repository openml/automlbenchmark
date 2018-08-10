import sys

import arff
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

train_data_path = "../common/train.arff"
test_data_path = "../common/test.arff"
output_path = "../common/predictions.csv"

runtime_seconds = sys.argv[1]
number_cores = int(sys.argv[2])
performance_metric = sys.argv[3]

print('Running RandomForest with a maximum time of {}s on {} cores.'.format(runtime_seconds, number_cores))
print('We completely ignore the requirement to stay within the time limit.')
print('We completely ignore the advice to optimize towards metric: {}.'.format(performance_metric))

with open(train_data_path, 'r') as train_file:
    train_data = arff.load(train_file)
    train_data_np = np.asarray(train_data['data'])
    X_train, y_train = train_data_np[:, :-1], train_data_np[:, -1]

with open(test_data_path, 'r') as test_file:
    test_data = arff.load(test_file)
    test_data_np = np.asarray(test_data['data'])
    X_test, y_test = test_data_np[:, :-1], test_data_np[:, -1]

print('Running model on task.')
rfc = RandomForestClassifier(n_jobs=number_cores)
rfc.fit(X_train, y_train)
class_predictions = rfc.predict(X_test).reshape(-1, 1)
class_probabilities = rfc.predict_proba(X_test)
combined_predictions = np.hstack((class_probabilities, class_predictions)).astype(str)

print(combined_predictions)
np.savetxt(output_path, combined_predictions, delimiter=',', fmt="%s")
