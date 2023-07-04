import logging
import os
import shutil
import warnings
import sys
import tempfile
import numpy as np
warnings.simplefilter("ignore")

if sys.platform == 'darwin':
    os.environ['OMP_NUM_THREADS'] = '1'

import pandas as pd

from autogluon.core.utils.savers import save_pd, save_pkl
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
from autogluon.timeseries.version import __version__

from frameworks.shared.callee import call_run, result, output_subdir
from frameworks.shared.utils import Timer, zip_path

log = logging.getLogger(__name__)


def run(dataset, config):
    log.info(f"\n**** AutoGluon TimeSeries [v{__version__}] ****\n")
    prediction_length = dataset.forecast_horizon_in_steps

    train_data = TimeSeriesDataFrame.from_path(
        dataset.train_path,
        id_column=dataset.id_column,
        timestamp_column=dataset.timestamp_column,
    )

    test_data = TimeSeriesDataFrame.from_path(
        dataset.test_path,
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
            quantile_levels=dataset.quantile_levels,
        )
        predictor.fit(
            train_data=train_data,
            time_limit=config.max_runtime_seconds,
            **{k: v for k, v in config.framework_params.items() if not k.startswith('_')},
        )

    with Timer() as predict:
        predictions = pd.DataFrame(predictor.predict(train_data))

    predictions_only = predictions['mean'].values
    test_data_future = test_data.slice_by_timestep(-prediction_length, None)
    assert test_data_future.index.equals(predictions.index), "Predictions and test data index do not match"
    truth_only = test_data_future[dataset.target].values

    leaderboard = predictor.leaderboard(test_data, silent=True)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        log.info(leaderboard)

    num_models_trained = len(leaderboard)

    save_artifacts(predictor=predictor, leaderboard=leaderboard, config=config)
    shutil.rmtree(predictor.path, ignore_errors=True)

    optional_columns = dict(
        repeated_item_id=np.load(dataset.repeated_item_id),
        repeated_abs_seasonal_error=np.load(dataset.repeated_abs_seasonal_error),
    )
    for q in dataset.quantile_levels:
        optional_columns[str(q)] = predictions[str(q)].values

    return result(output_file=config.output_predictions_file,
                  predictions=predictions_only,
                  truth=truth_only,
                  target_is_encoded=False,
                  models_count=num_models_trained,
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
