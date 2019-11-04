import logging
import os
import sys

import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

__path__ = [os.path.join(os.getcwd(), 'frameworks', 'shared'),
            os.path.join(os.getcwd(), 'amlb', 'utils')]
from .callee import call_run, result

log = logging.getLogger(__name__)


def run(dataset, config):
    print("YYYYYYY")
    log.info("\n**** Random Forest (sklearn %s) ****\n", sklearn.__version__)

    is_classification = config.type == 'classification'

    X_train, X_test = dataset.train.X_enc, dataset.test.X_enc
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

    return result(output_file=config.output_predictions_file,
                  predictions=predictions,
                  truth=y_test,
                  probabilities=probabilities,
                  target_is_encoded=False)


if __name__ == '__main__':
    call_run(run)
