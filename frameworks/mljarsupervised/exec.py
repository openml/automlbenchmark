import os
import shutil
import logging

import pandas as pd
import matplotlib
matplotlib.use("agg")  # no need for tk

import supervised
from supervised.automl import AutoML

from frameworks.shared.callee import call_run, result, output_subdir, save_metadata, utils

log = logging.getLogger(os.path.basename(__file__))

def run(dataset, config):
    log.info(f"\n**** mljar-supervised [v{supervised.__version__}] ****\n")
    save_metadata(config, version=supervised.__version__)

    column_names, _ = zip(*dataset.columns)
    column_types = dict(dataset.columns)
    X_train = pd.DataFrame(dataset.train.X, columns=column_names).astype(
        column_types, copy=False
    )
    X_test = pd.DataFrame(dataset.test.X, columns=column_names).astype(
        column_types, copy=False
    )

    y_train = dataset.train.y.flatten()
    y_test = dataset.test.y.flatten()

    problem_mapping = dict(
        binary="binary_classification",
        multiclass="multiclass_classification",
        regression="regression",
    )
    is_classification = config.type == "classification"
    ml_task = problem_mapping.get(
        dataset.problem_type
    )  # if None the AutoML will guess about the ML task
    results_path = output_subdir("results", config)
    training_params = {
        k: v for k, v in config.framework_params.items() if not k.startswith("_")
    }

    automl = AutoML(
        results_path=results_path,
        total_time_limit=config.max_runtime_seconds,
        seed=config.seed,
        ml_task=ml_task,
        **training_params
    )

    with utils.Timer() as training:
        automl.fit(X_train, y_train)

    with utils.Timer() as predict:
        preds = automl.predict(X_test)

    predictions, probabilities = None, None
    if is_classification:
        predictions = preds["label"].values
        probabilities = preds[preds.columns[:-1]].values
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
        predict_duration=predict.duration
    )


if __name__ == "__main__":
    call_run(run)
