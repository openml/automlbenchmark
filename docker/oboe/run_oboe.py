import sys

sys.path.append('../common')
import common_code

sys.path.append('/bench/oboe/automl')
from auto_learner import AutoLearner
from sklearn.preprocessing import LabelEncoder

runtime_seconds = int(sys.argv[1])
number_cores = int(sys.argv[2])
performance_metric = sys.argv[3]

X_train, y_train = common_code.get_X_y_from_arff(common_code.TRAIN_DATA_PATH)
X_test, y_test = common_code.get_X_y_from_arff(common_code.TEST_DATA_PATH)
X_train, X_test = X_train.astype(float), X_test.astype(float)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)


print('Running oboe with a maximum time of {}s on {} cores.'.format(runtime_seconds, number_cores))
print('We completely ignore the advice to optimize towards metric: {}.'.format(performance_metric))

automl = AutoLearner(p_type='classification',
                     n_cores=number_cores,
                     runtime_limit=runtime_seconds
                     )
automl.fit_doubling_time_constrained(X_train, y_train)
class_predictions = automl.predict(X_test)
class_predictions = le.inverse_transform(class_predictions).reshape(-1, 1)
class_probabilities = common_code.one_hot_encode_predictions(class_predictions)

common_code.save_predictions_to_file(class_probabilities, class_predictions)

