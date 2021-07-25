import os
import shutil
import logging

import numpy as np
import matplotlib
matplotlib.use("agg")  # no need for tk

import supervised
from supervised.automl import AutoML

from frameworks.shared.callee import call_run, result, output_subdir
from frameworks.shared.utils import Timer

log = logging.getLogger(os.path.basename(__file__))


def run(dataset, config):
    log.info(f"\n**** mljar-supervised [v{supervised.__version__}] ****\n")

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

    X_train, y_train = dataset.train.X, dataset.train.y.squeeze()
    X_test, y_test = dataset.test.X, dataset.test.y.squeeze()

    automl = AutoML(
        results_path=results_path,
        total_time_limit=config.max_runtime_seconds,
        random_state=config.seed,
        ml_task=ml_task,
        eval_metric=eval_metric,
        **training_params
    )

    with Timer() as training:
        automl.fit(X_train, y_train)

    with Timer() as predict:
        preds = automl.predict_all(X_test)

    predictions, probabilities, probabilities_labels = None, None, None
    if is_classification:
        # preds is a dataframe with columns ["prediction_LABEL", .., "label"]
        if y_train.dtype == bool and preds["label"].dtype == int:
            # boolean target produces integer predictions for mljar-supervised <= 0.10.6
            # https://github.com/mljar/mljar-supervised/issues/442
            preds = preds.rename({"prediction_0": "False", "prediction_1": "True"}, axis=1)
            preds["label"] = preds["label"].astype(bool)
        else:
            preds.columns = [c.lstrip("prediction_") for c in preds.columns]
        
        predictions = preds["label"].values
        probabilities_labels = list(preds.columns)[:-1]
        probabilities = preds[probabilities_labels].values
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
        probabilities_labels=probabilities_labels,
        models_count=len(automl._models),
        training_duration=training.duration,
        predict_duration=predict.duration
    )


if __name__ == "__main__":
    call_run(run)
