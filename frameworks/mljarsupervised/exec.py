import os
import shutil
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("agg")  # no need for tk

from supervised.automl import AutoML

from frameworks.shared.callee import call_run, result, output_subdir, utils

log = logging.getLogger(os.path.basename(__file__))

def run(dataset, config):
    log.info("\n**** mljar-supervised ****\n")

    # Mapping of benchmark metrics to MLJAR metrics
    metrics_mapping = dict(
        auc='auc',
        logloss='logloss',
        rmse='rmse'
    )
    eval_metric = metrics_mapping[config.metric] if config.metric in metrics_mapping else "auto"
    
    # Mapping of benchmark task to MLJAR ML task
    problem_mapping = dict(
        binary="binary_classification",
        multiclass="multiclass_classification",
        regression="regression",
    )
    ml_task = problem_mapping.get(
        dataset.problem_type
    )  # if None the AutoML will guess about the ML task
    is_classification = config.type == "classification"
    results_path = output_subdir("results", config)
    training_params = {
        k: v for k, v in config.framework_params.items() if not k.startswith("_")
    }

    column_names, _ = zip(*dataset.columns)
    column_types = dict(dataset.columns)
    label = dataset.target.name

    train = pd.DataFrame(dataset.train.data, columns=column_names).astype(column_types, copy=False)
    X_train = train.drop(columns=label)
    y_train = train[label]

    test = pd.DataFrame(dataset.test.data, columns=column_names).astype(column_types, copy=False)
    X_test = test.drop(columns=label)
    y_test = test[label]

    automl = AutoML(
        results_path=results_path,
        total_time_limit=config.max_runtime_seconds,
        random_state=config.seed,
        ml_task=ml_task,
        eval_metric=eval_metric,
        **training_params
    )

    with utils.Timer() as training:
        automl.fit(X_train, y_train)

    preds = automl.predict_all(X_test)

    predictions, probabilities = None, None
    if is_classification:
        predictions = preds["label"].values
        cols = [f"prediction_{c}" for c in np.unique(y_train)]
        probabilities = preds[cols].values
    else:
        predictions = preds["prediction"].values

    # clean the results
    if not config.framework_params.get("_save_artifacts", False):
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
