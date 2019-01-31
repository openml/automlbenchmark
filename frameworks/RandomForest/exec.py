import logging

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from automl.benchmark import TaskConfig
from automl.data import Dataset
from automl.datautils import impute
from automl.results import save_predictions_to_file

log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    log.info("\n**** Random Forest (sklearn) ****\n")

    is_classification = config.type == 'classification'

    # Impute any missing data (can test using -t 146606)
    X_train, X_test = impute(dataset.train.X_enc, dataset.test.X_enc)
    y_train, y_test = dataset.train.y, dataset.test.y

    log.info("Running RandomForest with a maximum time of {}s on {} cores.".format(config.max_runtime_seconds, config.cores))
    log.warning("We completely ignore the requirement to stay within the time limit.")
    log.warning("We completely ignore the advice to optimize towards metric: {}.".format(config.metric))

    estimator = RandomForestClassifier if is_classification else RandomForestRegressor
    rfc = estimator(n_jobs=config.cores,
                    **config.framework_params)

    rfc.fit(X_train, y_train)

    predictions = rfc.predict(X_test)
    probabilities = rfc.predict_proba(X_test) if is_classification else None

    save_predictions_to_file(dataset=dataset,
                             output_file=config.output_predictions_file,
                             probabilities=probabilities,
                             predictions=predictions,
                             truth=y_test,
                             target_is_encoded=False)

