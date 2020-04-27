import logging
import os
import tempfile as tmp
import pandas as pd
import numpy as np

import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from frameworks.shared.callee import call_run, result, Timer

log = logging.getLogger(os.path.basename(__file__))

import supervised
from supervised.automl import AutoML
import shutil

def run(dataset, config):
    log.info("\n**** mljar-supervised ****\n")

    is_classification = config.type == "classification"

    X_train, X_test = dataset.train.X, dataset.test.X
    y_train, y_test = dataset.train.y, dataset.test.y

    X_train = pd.DataFrame(X_train, columns=[f"f{i}" for i in range(X_train.shape[1])])
    X_test = pd.DataFrame(X_test, columns=[f"f{i}" for i in range(X_test.shape[1])])

    y_train = y_train.flatten()
    y_test = y_test.flatten()


    # cast columns to have float types
    for col in X_train.columns:
        try: 
            x1 = X_train[col].astype(float)
            X_train[col] = x1
            x2 = X_test[col].astype(float)
            X_test[col] = x2
        except Exception as e:
            pass

    print(
        "We completely ignore the advice to optimize towards metric: {}.".format(
            config.metric
        )
    )
    
    ml_task = None  # the AutoML will guess about ML task
    # however, in case of multiclass classification, we will set the ml_task manually.
    # AutoML assumes that multiclass calssification is when there is from 2 to 20 unique values.
    # In this benchmark, there can be casses when it is much more classes than 20. That's why we set manually.
    if is_classification and len(np.unique(y_train)) > 2:
        ml_task = "multiclass_classification"

    algorithms = ["Baseline", "Decision Tree", "Random Forest", "Xgboost", "LightGBM"]
    if config.max_runtime_seconds <= 60:
        algorithms = ["Baseline"]
    results_path = "AutoML_results"
    shutil.rmtree(results_path, ignore_errors=True) # clear just in case
    automl = AutoML(
        results_path=results_path,
        algorithms=algorithms,
        total_time_limit=config.max_runtime_seconds,
        seed=12,
        ml_task=ml_task
    )
    
    with Timer() as training:
        automl.fit(X_train, y_train)

    preds = automl.predict(X_test)

    predictions, probabilities = None, None
    if is_classification:
        predictions = preds["label"].values
        probabilities = preds[preds.columns[:-1]].values
    else:
        predictions = preds["prediction"].values

    # clean the results 
    # if you want to inspect results, please comment below line
    shutil.rmtree(results_path, ignore_errors=True)

    return result(
        output_file=config.output_predictions_file,
        predictions=predictions,
        truth=y_test,
        probabilities=probabilities,
        models_count=len(automl._models),
        training_duration=training.duration,
    )


if __name__ == "__main__":
    call_run(run)