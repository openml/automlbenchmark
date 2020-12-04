import logging
import os
import tempfile as tmp

os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from frameworks.shared.callee import call_run, result, save_metadata, utils

log = logging.getLogger(os.path.basename(__file__))


def run(dataset, config):
    log.info(f"\n**** Random Forest [sklearn v{sklearn.__version__}] ****\n")
    save_metadata(config, version=sklearn.__version__)

    is_classification = config.type == 'classification'

    X_train, X_test = dataset.train.X_enc, dataset.test.X_enc
    y_train, y_test = dataset.train.y_enc, dataset.test.y_enc

    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}
    n_jobs = config.framework_params.get('_n_jobs', config.cores)  # useful to disable multicore, regardless of the dataset config

    log.info("Running RandomForest with a maximum time of {}s on {} cores.".format(config.max_runtime_seconds, n_jobs))
    log.warning("We completely ignore the requirement to stay within the time limit.")
    log.warning("We completely ignore the advice to optimize towards metric: {}.".format(config.metric))

    estimator = RandomForestClassifier if is_classification else RandomForestRegressor
    rf = estimator(n_jobs=n_jobs,
                   random_state=config.seed,
                   **training_params)

    with utils.Timer() as training:
        rf.fit(X_train, y_train)

    predictions = rf.predict(X_test)
    probabilities = rf.predict_proba(X_test) if is_classification else None

    return result(output_file=config.output_predictions_file,
                  predictions=predictions,
                  truth=y_test,
                  probabilities=probabilities,
                  target_is_encoded=is_classification,
                  models_count=len(rf),
                  training_duration=training.duration)


if __name__ == '__main__':
    call_run(run)
