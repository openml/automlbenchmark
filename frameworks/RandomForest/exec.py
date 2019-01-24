import logging

from sklearn.ensemble import RandomForestClassifier

from automl.benchmark import TaskConfig
from automl.data import Dataset
from automl.datautils import impute
from automl.results import save_predictions_to_file

log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    log.info("\n**** Random Forest (sklearn) ****\n")

    # Impute any missing data (can test using -t 146606)
    X_train, X_test = impute(dataset.train.X_enc, dataset.test.X_enc)
    y_train, y_test = dataset.train.y, dataset.test.y

    log.info('Running RandomForest with a maximum time of {}s on {} cores.'.format(config.max_runtime_seconds, config.cores))
    log.warning('We completely ignore the requirement to stay within the time limit.')
    log.warning('We completely ignore the advice to optimize towards metric: {}.'.format(config.metric))

    rfc = RandomForestClassifier(n_jobs=config.cores, **config.framework_params)
    rfc.fit(X_train, y_train)
    class_predictions = rfc.predict(X_test)
    class_probabilities = rfc.predict_proba(X_test)

    save_predictions_to_file(dataset=dataset,
                             output_file=config.output_predictions_file,
                             class_probabilities=class_probabilities,
                             class_predictions=class_predictions,
                             class_truth=y_test,
                             classes_are_encoded=False)

