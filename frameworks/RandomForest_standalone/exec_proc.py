import copy
import logging
import os
import sys

import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

sys.path.append(".")
from automl.datautils import impute, read_csv, write_csv
from automl.utils import json_dumps, json_loads, Namespace as ns

log = logging.getLogger(__name__)


def run(dataset, config):
    log.info("\n**** Random Forest (sklearn %s) ****\n", sklearn.__version__)

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

    return ns(output_file=config.output_predictions_file,
              probabilities=probabilities,
              predictions=predictions,
              truth=y_test,
              target_is_encoded=False)


if __name__ == '__main__':
    params = json_loads(sys.stdin.read(), as_namespace=True)

    def load_data(path):
        return read_csv(path, as_data_frame=False, header=False)

    ds = ns(
        train=ns(
            X_enc=load_data(params.dataset.train.X_enc),
            y=load_data(params.dataset.train.y).squeeze()
        ),
        test=ns(
            X_enc=load_data(params.dataset.test.X_enc),
            y=load_data(params.dataset.test.y).squeeze(),
        )
    )
    config = params.config
    config.framework_params = ns.dict(config.framework_params)
    result = run(ds, config)

    res = copy.copy(result)
    res.predictions = os.path.join(config.result_dir, 'predictions')
    res.truth = os.path.join(config.result_dir, 'truth')
    write_csv(result.predictions.reshape(-1, 1), res.predictions)
    write_csv(result.truth.reshape(-1, 1), res.truth)
    if result.probabilities is not None:
        res.probabilities = os.path.join(config.result_dir, 'probabilities')
        write_csv(result.probabilities, res.probabilities)

    print(config.result_token)
    print(json_dumps(res, style='compact'))
