from collections import defaultdict
import sys

import numpy as np
sys.path.append('../common')
import common_code

runtime_seconds = sys.argv[1]
number_cores = int(sys.argv[2])
performance_metric = sys.argv[3]

X_train, y_train = common_code.get_X_y_from_arff(common_code.TRAIN_DATA_PATH)
X_test, y_test = common_code.get_X_y_from_arff(common_code.TEST_DATA_PATH)

# collections.Counter does not keep insertion order.
counter = defaultdict(int)
for cls in y_train:
    counter[cls] += 1

N_train = len(y_train)
N_test = len(y_test)

class_probabilities = [class_count/N_train for (class_, class_count) in counter.items()]
majority_class = max(counter.items(), key=lambda k_v: k_v[1])[0]
class_probabilities = np.tile(class_probabilities, reps=(N_test, 1))
class_prediction = np.tile(majority_class, reps=(N_test,))

common_code.save_predictions_to_file(class_probabilities, class_prediction)