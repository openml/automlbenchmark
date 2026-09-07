import logging
import os
import tempfile as tmp

os.environ["JOBLIB_TEMP_FOLDER"] = tmp.gettempdir()
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sklearn
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.svm import LinearSVR

from frameworks.shared.callee import call_run, result
from frameworks.shared.utils import Timer

log = logging.getLogger(os.path.basename(__file__))


def run(dataset, config):
    log.info(f"\n**** Stacking Ensemble [sklearn v{sklearn.__version__}] ****\n")

    is_classification = config.type == "classification"

    X_train, X_test = dataset.train.X_enc, dataset.test.X_enc
    y_train, y_test = dataset.train.y_enc, dataset.test.y_enc

    training_params = {
        k: v for k, v in config.framework_params.items() if not k.startswith("_")
    }
    n_jobs = config.framework_params.get(
        "_n_jobs", config.cores
    )  # useful to disable multicore, regardless of the dataset config
    estimators_params = {
        e: config.framework_params.get(f"_{e}_params", {})
        for e in ["rf", "gbm", "sgdclassifier", "sgdregressor", "svc", "final"]
    }

    log.info(
        f"Running Sklearn Stacking Ensemble with a maximum time of {config.max_runtime_seconds}s on {n_jobs} cores."
    )
    log.warning("We completely ignore the requirement to stay within the time limit.")
    log.warning(
        f"We completely ignore the advice to optimize towards metric: {config.metric}."
    )

    if is_classification:
        estimator = StackingClassifier(
            estimators=[
                (
                    "rf",
                    RandomForestClassifier(
                        n_jobs=n_jobs,
                        random_state=config.seed,
                        **estimators_params["rf"],
                    ),
                ),
                (
                    "gbm",
                    GradientBoostingClassifier(
                        random_state=config.seed, **estimators_params["gbm"]
                    ),
                ),
                (
                    "linear",
                    SGDClassifier(
                        n_jobs=n_jobs,
                        random_state=config.seed,
                        **estimators_params["sgdclassifier"],
                    ),
                ),
                # ('svc', LinearSVC(random_state=config.seed, **estimators_params['svc']))
            ],
            # final_estimator=SGDClassifier(n_jobs=n_jobs, random_state=config.seed, **estimators_params['final']),
            final_estimator=LogisticRegression(
                n_jobs=n_jobs, random_state=config.seed, **estimators_params["final"]
            ),
            stack_method="predict_proba",
            n_jobs=n_jobs,
            **training_params,
        )
    else:
        estimator = StackingRegressor(
            estimators=[
                (
                    "rf",
                    RandomForestRegressor(
                        n_jobs=n_jobs,
                        random_state=config.seed,
                        **estimators_params["rf"],
                    ),
                ),
                (
                    "gbm",
                    GradientBoostingRegressor(
                        random_state=config.seed, **estimators_params["gbm"]
                    ),
                ),
                (
                    "linear",
                    SGDRegressor(
                        random_state=config.seed, **estimators_params["sgdregressor"]
                    ),
                ),
                (
                    "svc",
                    LinearSVR(random_state=config.seed, **estimators_params["svc"]),
                ),
            ],
            # final_estimator=SGDRegressor(random_state=config.seed, **estimators_params['final']),
            final_estimator=LinearRegression(n_jobs=n_jobs),
            n_jobs=n_jobs,
            **training_params,
        )

    with Timer() as training:
        estimator.fit(X_train, y_train)

    with Timer() as predict:
        predictions = estimator.predict(X_test)
    probabilities = estimator.predict_proba(X_test) if is_classification else None

    return result(
        output_file=config.output_predictions_file,
        predictions=predictions,
        truth=y_test,
        probabilities=probabilities,
        target_is_encoded=is_classification,
        models_count=len(estimator.estimators_) + 1,
        training_duration=training.duration,
        predict_duration=predict.duration,
    )


if __name__ == "__main__":
    call_run(run)
