import logging
import os
import shutil
import warnings
import gc
from io import StringIO
warnings.simplefilter("ignore")

import matplotlib
import pandas as pd
matplotlib.use('agg')  # no need for tk

from autogluon.tabular import TabularPredictor
from autogluon.core.utils.savers import save_pd, save_pkl
import autogluon.core.metrics as metrics
from autogluon.tabular.version import __version__

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
        rmse=metrics.root_mean_squared_error,
    )

    label = dataset.target.name
    problem_type = dataset.problem_type

    perf_metric = metrics_mapping[config.metric] if config.metric in metrics_mapping else None
    if perf_metric is None:
        # TODO: figure out if we are going to blindly pass metrics through, or if we use a strict mapping
        log.warning("Performance metric %s not supported.", config.metric)

    is_classification = config.type == 'classification'
    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}

    load_raw = config.framework_params.get('_load_raw', False)
    if load_raw:
        train, test = load_data_raw(dataset=dataset)
    else:
        column_names, _ = zip(*dataset.columns)
        column_types = dict(dataset.columns)
        train = pd.DataFrame(dataset.train.data, columns=column_names).astype(column_types, copy=False)
        print(f"Columns dtypes:\n{train.dtypes}")
        test = pd.DataFrame(dataset.test.data, columns=column_names).astype(column_types, copy=False)

    del dataset
    gc.collect()

    output_dir = output_subdir("models", config)
    with utils.Timer() as training:
        predictor = TabularPredictor(
            label=label,
            eval_metric=perf_metric.name,
            path=output_dir,
            problem_type=problem_type,
        ).fit(
            train_data=train,
            time_limit=config.max_runtime_seconds,
            **training_params
        )

    del train

    y_test = test[label]
    test = test.drop(columns=label)

    if is_classification:
        with utils.Timer() as predict:
            probabilities = predictor.predict_proba(test, as_multiclass=True)
        predictions = probabilities.idxmax(axis=1).to_numpy()
    else:
        with utils.Timer() as predict:
            predictions = predictor.predict(test, as_pandas=False)
        probabilities = None

    prob_labels = probabilities.columns.values.tolist() if probabilities is not None else None

    leaderboard = predictor.leaderboard(silent=True)  # Removed test data input to avoid long running computation, remove 7200s timeout limitation to re-enable
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


def load_data_raw(dataset):
    """
    Save and load data while removing all preset dtype information.
    This represents the most challenging scenario of reading from raw CSV without dtype information.
    """
    label = dataset.target.name
    column_names, _ = zip(*dataset.columns)
    train = pd.DataFrame(dataset.train.data, columns=column_names).infer_objects()
    test = pd.DataFrame(dataset.test.data, columns=column_names).infer_objects()

    # Save and load data to remove any pre-set dtypes, observe performance from worst-case scenario: raw csv
    train = convert_to_raw(train, label=label)
    test = convert_to_raw(test, label=label)

    return train, test


# Remove custom type information
def convert_to_raw(X, label=None):
    if label is not None:
        y = X[label]
        X = X.drop(columns=[label])
    else:
        y = None
    with StringIO() as buffer:
        X.to_csv(buffer, index=True, header=True)
        buffer.seek(0)
        X = pd.read_csv(buffer, index_col=0, header=0, low_memory=False, encoding='utf-8')
    if label is not None:
        X[label] = y
    return X


if __name__ == '__main__':
    call_run(run)
