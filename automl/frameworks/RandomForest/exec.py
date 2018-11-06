import os

from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier

from automl.benchmark import TaskConfig
from automl.data import Dataset
from automl.utils import save_predictions_to_file

def run(dataset: Dataset, config: TaskConfig):
    print("\n**** Random Forest (sklearn) ****\n")

    # Impute any missing data (can test using -t 146606)
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(dataset.train.X_enc)
    X_train = imp.transform(dataset.train.X_enc)
    y_train = dataset.train.y_enc
    X_test = imp.transform(dataset.test.X_enc)

    # TODO: Probably have to add a dummy encoder here in case there's any categoricals
    # TODO: If auto-sklearn & TPOT also require imputation & dummy encoding, let's move this to common_code

    print('Running RandomForest with a maximum time of {}s on {} cores.'.format(config.max_runtime_seconds, config.cores))
    print('We completely ignore the requirement to stay within the time limit.')
    print('We completely ignore the advice to optimize towards metric: {}.'.format(config.metric))

    rfc = RandomForestClassifier(n_jobs=config.cores, n_estimators=2000)
    rfc.fit(X_train, y_train)
    class_predictions = rfc.predict(X_test)
    class_probabilities = rfc.predict_proba(X_test)

    #todo: accuracy

    dest_file = os.path.join(os.path.expanduser(config.output_folder), "predictions_random_forest_{task}_{fold}.txt".format(task=config.name, fold=config.fold))
    save_predictions_to_file(class_probabilities, class_predictions, dest_file)
    print("Predictions saved to "+dest_file)
    print()
