import logging
import numpy as np
import os
import pandas as pd
import shutil
import sys
import tempfile
import warnings
warnings.simplefilter("ignore")

if sys.platform == 'darwin':
    os.environ['OMP_NUM_THREADS'] = '1'

from autogluon.core.utils.savers import save_pd, save_pkl
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
from autogluon.timeseries.version import __version__
from joblib.externals.loky import get_reusable_executor

from frameworks.shared.callee import call_run, result, output_subdir
from frameworks.shared.utils import Timer, zip_path, load_timeseries_dataset

log = logging.getLogger(__name__)


def run(dataset, config):
    log.info(f"\n**** AutoGluon TimeSeries [v{__version__}] ****\n")
    prediction_length = dataset.forecast_horizon_in_steps
    train_df, test_df = load_timeseries_dataset(dataset)

    train_data = TimeSeriesDataFrame.from_data_frame(
        train_df,
        id_column=dataset.id_column,
        timestamp_column=dataset.timestamp_column,
    )

    test_data = TimeSeriesDataFrame.from_data_frame(
        test_df,
        id_column=dataset.id_column,
        timestamp_column=dataset.timestamp_column,
    )

    predictor_path = tempfile.mkdtemp() + os.sep
    with Timer() as training:
        predictor = TimeSeriesPredictor(
            target=dataset.target,
            path=predictor_path,
            prediction_length=prediction_length,
            eval_metric=get_eval_metric(config),
            eval_metric_seasonal_period=dataset.seasonality,
            quantile_levels=config.quantile_levels,
        )
        predictor.fit(
            train_data=train_data,
            time_limit=config.max_runtime_seconds,
            random_seed=config.seed,
            **{k: v for k, v in config.framework_params.items() if not k.startswith('_')},
        )

    with Timer() as predict:
        predictions = pd.DataFrame(predictor.predict(train_data))

    # Add columns necessary for the metric computation + quantile forecast to `optional_columns`
    optional_columns = dict(
        repeated_item_id=np.load(dataset.repeated_item_id),
        repeated_abs_seasonal_error=np.load(dataset.repeated_abs_seasonal_error),
    )
    for q in config.quantile_levels:
        optional_columns[str(q)] = predictions[str(q)].values

    predictions_only = get_point_forecast(predictions, config.metric)
    truth_only = test_df[dataset.target].values

    # Sanity check - make sure predictions are ordered correctly
    assert predictions.index.equals(test_data.index), "Predictions and test data index do not match"

    test_data_full = pd.concat([train_data, test_data])
    leaderboard = predictor.leaderboard(test_data_full, silent=True)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        log.info(leaderboard)

    save_artifacts(predictor=predictor, leaderboard=leaderboard, config=config)
    shutil.rmtree(predictor.path, ignore_errors=True)

    # Kill child processes spawned by Joblib to avoid spam in the AMLB log
    get_reusable_executor().shutdown(wait=True)

    return result(output_file=config.output_predictions_file,
                  predictions=predictions_only,
                  truth=truth_only,
                  target_is_encoded=False,
                  models_count=len(leaderboard),
                  training_duration=training.duration,
                  predict_duration=predict.duration,
                  optional_columns=pd.DataFrame(optional_columns))


def get_eval_metric(config):
    # TODO: Support more metrics
    metrics_mapping = dict(
        mape="MAPE",
        smape="sMAPE",
        mase="MASE",
        mse="MSE",
        rmse="RMSE",
    )

    eval_metric = metrics_mapping[config.metric] if config.metric in metrics_mapping else None
    if eval_metric is None:
        log.warning("Performance metric %s not supported.", config.metric)
    return eval_metric


def get_point_forecast(predictions, metric):
    # Return median for metrics optimized by median, if possible
    if metric.lower() in ["rmse", "mse"] or "0.5" not in predictions.columns:
        log.info("Using mean as point forecast")
        return predictions["mean"].values
    else:
        log.info("Using median as point forecast")
        return predictions["0.5"].values


def save_artifacts(predictor, leaderboard, config):
    artifacts = config.framework_params.get('_save_artifacts', ['leaderboard'])
    try:
        if 'leaderboard' in artifacts:
            leaderboard_dir = output_subdir("leaderboard", config)
            save_pd.save(path=os.path.join(leaderboard_dir, "leaderboard.csv"), df=leaderboard)

        if 'info' in artifacts:
            ag_info = predictor.info()
            info_dir = output_subdir("info", config)
            save_pkl.save(path=os.path.join(info_dir, "info.pkl"), object=ag_info)

        if 'models' in artifacts:
            shutil.rmtree(os.path.join(predictor.path, "utils"), ignore_errors=True)
            models_dir = output_subdir("models", config)
            zip_path(predictor.path, os.path.join(models_dir, "models.zip"))
    except Exception:
        log.warning("Error when saving artifacts.", exc_info=True)


if __name__ == '__main__':
    call_run(run)
