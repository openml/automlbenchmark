import logging
import os

from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier

from automl.benchmark import TaskConfig
from automl.data import Dataset
from automl.results import save_predictions_to_file

log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    log.info("\n**** Random Forest (sklearn) ****\n")

    # Impute any missing data (can test using -t 146606)
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(dataset.train.X_enc)
    X_train = imp.transform(dataset.train.X_enc)
    y_train = dataset.train.y_enc
    X_test = imp.transform(dataset.test.X_enc)
    y_test = dataset.test.y_enc

    # TODO: Probably have to add a dummy encoder here in case there's any categoricals
    # TODO: If auto-sklearn & TPOT also require imputation & dummy encoding, let's move this to common_code

    log.info('Running RandomForest with a maximum time of {}s on {} cores.'.format(config.max_runtime_seconds, config.cores))
    log.warning('We completely ignore the requirement to stay within the time limit.')
    log.warning('We completely ignore the advice to optimize towards metric: {}.'.format(config.metric))

    rfc = RandomForestClassifier(n_jobs=config.cores, n_estimators=2000)
    rfc.fit(X_train, y_train)
    class_predictions = rfc.predict(X_test)
    class_probabilities = rfc.predict_proba(X_test)

    save_predictions_to_file(dataset=dataset,
                             output_file=config.output_file_template,
                             class_probabilities=class_probabilities,
                             class_predictions=class_predictions,
                             class_truth=y_test)

