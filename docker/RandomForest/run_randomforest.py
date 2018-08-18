import sys

from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier

sys.path.append('/bench/common')
import common_code

runtime_seconds = sys.argv[1]
number_cores = int(sys.argv[2])
performance_metric = sys.argv[3]

X_train, y_train = common_code.get_X_y_from_arff(common_code.TRAIN_DATA_PATH)
X_test, y_test = common_code.get_X_y_from_arff(common_code.TEST_DATA_PATH)

# Impute any missing data (can test using -t 146606)
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(X_train)
X_train = imp.transform(X_train)
X_test = imp.transform(X_test)

# TO DO: Probably have to add a dummy encoder here in case there's any categoricals
# TO DO: If auto-sklearn & TPOT also require imputation & dummy encoding, let's move this to common_code 

print('Running RandomForest with a maximum time of {}s on {} cores.'.format(runtime_seconds, number_cores))
print('We completely ignore the requirement to stay within the time limit.')
print('We completely ignore the advice to optimize towards metric: {}.'.format(performance_metric))

rfc = RandomForestClassifier(n_jobs=number_cores, n_estimators=100)
rfc.fit(X_train, y_train)
class_predictions = rfc.predict(X_test)
class_probabilities = rfc.predict_proba(X_test)

common_code.save_predictions_to_file(class_probabilities, class_predictions)

