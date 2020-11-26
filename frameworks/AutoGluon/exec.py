import logging
import os
import shutil
import warnings
warnings.simplefilter("ignore")

import matplotlib
import pandas as pd
matplotlib.use('agg')  # no need for tk

from autogluon.task.tabular_prediction.tabular_prediction import TabularPrediction as task
from autogluon.utils.tabular.utils.savers import save_pd, save_pkl
import autogluon.utils.tabular.metrics as metrics
from autogluon.version import __version__

from frameworks.shared.callee import call_run, result, output_subdir, utils, save_metadata

log = logging.getLogger(__name__)


def run(dataset, config):
    log.info(f"\n**** AutoGluon [v{__version__}] ****\n")
    save_metadata(config, version=__version__)

    metrics_mapping = dict(
        acc=metrics.accuracy,
        auc=metrics.roc_auc,
        f1=metrics.f1,
        logloss=metrics.log_loss,
        mae=metrics.mean_absolute_error,
        mse=metrics.mean_squared_error,
        r2=metrics.r2,
        # rmse=metrics.root_mean_squared_error,  # metrics.root_mean_squared_error incorrectly registered in autogluon REGRESSION_METRICS
        rmse=metrics.mean_squared_error,  # for now, we can let autogluon optimize training on mse: anyway we compute final score from predictions.
    )

    perf_metric = metrics_mapping[config.metric] if config.metric in metrics_mapping else None
    if perf_metric is None:
        # TODO: figure out if we are going to blindly pass metrics through, or if we use a strict mapping
        log.warning("Performance metric %s not supported.", config.metric)

    is_classification = config.type == 'classification'
    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}

    column_names, _ = zip(*dataset.columns)
    column_types = dict(dataset.columns)
    train = pd.DataFrame(dataset.train.data, columns=column_names).astype(column_types, copy=False)
    label = dataset.target.name
    print(f"Columns dtypes:\n{train.dtypes}")

    output_dir = output_subdir("models", config)
    with utils.Timer() as training:
        predictor = task.fit(
            train_data=train,
            label=label,
            problem_type=dataset.problem_type,
            output_directory=output_dir,
            time_limits=config.max_runtime_seconds,
            eval_metric=perf_metric.name,
            **training_params
        )

    test = pd.DataFrame(dataset.test.data, columns=column_names).astype(column_types, copy=False)
    X_test = test.drop(columns=label)
    y_test = test[label]

    with utils.Timer() as predict:
        predictions = predictor.predict(X_test)

    probabilities = predictor.predict_proba(dataset=X_test, as_pandas=True, as_multiclass=True) if is_classification else None
    prob_labels = probabilities.columns.values.tolist() if probabilities is not None else None

    leaderboard = predictor._learner.leaderboard(X_test, y_test, silent=True)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(leaderboard)

    save_artifacts(predictor, leaderboard, config)

    num_models_trained = len(leaderboard)
    num_models_ensemble = len(predictor._trainer.get_minimum_model_set(predictor._trainer.model_best))

    return result(output_file=config.output_predictions_file,
                  predictions=predictions,
                  truth=y_test,
                  probabilities=probabilities,
                  probabilities_labels=prob_labels,
                  target_is_encoded=False,
                  models_count=num_models_trained,
                  models_ensemble_count=num_models_ensemble,
                  training_duration=training.duration,
                  predict_duration=predict.duration)


def save_artifacts(predictor, leaderboard, config):
    artifacts = config.framework_params.get('_save_artifacts', ['leaderboard'])
    try:
        models_dir = output_subdir("models", config)
        shutil.rmtree(os.path.join(models_dir, "utils"), ignore_errors=True)

        if 'leaderboard' in artifacts:
            save_pd.save(path=os.path.join(models_dir, "leaderboard.csv"), df=leaderboard)

        if 'info' in artifacts:
            ag_info = predictor.info()
            info_dir = output_subdir("info", config)
            save_pkl.save(path=os.path.join(info_dir, "info.pkl"), object=ag_info)

        if 'models' in artifacts:
            utils.zip_path(models_dir,
                           os.path.join(models_dir, "models.zip"))

        def delete(path, isdir):
            if isdir:
                shutil.rmtree(path, ignore_errors=True)
            elif os.path.splitext(path)[1] == '.pkl':
                os.remove(path)
        utils.walk_apply(models_dir, delete, max_depth=0)

    except Exception:
        log.warning("Error when saving artifacts.", exc_info=True)


if __name__ == '__main__':
    call_run(run)
