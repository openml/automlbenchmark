import sys
from sklearn.dummy import DummyClassifier

sys.path.append('../common')
import common_code

runtime_seconds = sys.argv[1]
number_cores = int(sys.argv[2])
performance_metric = sys.argv[3]

X_train, y_train = common_code.get_X_y_from_arff(common_code.TRAIN_DATA_PATH)
X_test, y_test = common_code.get_X_y_from_arff(common_code.TEST_DATA_PATH)

dummy = DummyClassifier(strategy='prior')
dummy.fit(X_train, y_train)
class_probabilities = dummy.predict_proba(X_test)
class_predictions = dummy.predict(X_test)
common_code.save_predictions_to_file(class_probabilities, class_predictions)
