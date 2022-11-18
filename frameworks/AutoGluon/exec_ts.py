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
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame, TimeSeriesEvaluator
from autogluon.timeseries.version import __version__

from frameworks.shared.callee import call_run, result, output_subdir
from frameworks.shared.utils import Timer, zip_path

log = logging.getLogger(__name__)


def run(dataset, config):
    log.info(f"\n**** AutoGluon TimeSeries [v{__version__}] ****\n")

    timestamp_column = dataset.timestamp_column
    id_column = dataset.id_column
    prediction_length = dataset.forecast_horizon_in_steps

    eval_metric = get_eval_metric(config)
    target_column = dataset.target.name
    time_limit = config.max_runtime_seconds
    time_limit = 10.

    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}

    train_data, test_data = load_data(train_path=dataset.train.path,
                                      test_path=dataset.test.path,
                                      timestamp_column=timestamp_column,
                                      id_column=id_column)
    freq=test_data.freq
    seasonality = get_seasonality_from_freq(freq)

    test_data_past = test_data.copy().slice_by_timestep(slice(None, -prediction_length))

    predictor_path = tempfile.mkdtemp() + os.sep
    with Timer() as training:
        predictor = TimeSeriesPredictor(
            target=target_column,
            path=predictor_path,
            prediction_length=prediction_length,
            eval_metric=eval_metric,
        )
        predictor.fit(
            train_data=train_data,
            time_limit=time_limit,
            **training_params,
        )

    with Timer() as predict:
        predictions = predictor.predict(test_data_past)

    predictions_only = predictions['mean'].values
    test_data_future = test_data.copy().slice_by_timestep(slice(-prediction_length, None))
    truth_only = test_data_future[target_column].values

    leaderboard = predictor.leaderboard(test_data, silent=True)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        log.info(leaderboard)

    num_models_trained = len(leaderboard)

    save_artifacts(predictor=predictor, leaderboard=leaderboard, config=config, test_data=test_data, predictions=predictions)
    shutil.rmtree(predictor.path, ignore_errors=True)

    quantiles = predictions.drop(columns=['mean']).reset_index(drop=True)

    forecast_unique_item_ids = np.arange(predictions_only.shape[0] / prediction_length)
    forecast_item_ids = np.repeat(forecast_unique_item_ids, prediction_length)

    seasonal_error_rep = calc_seasonal_error(dataset_test=test_data, id_column='item_id',
                                             target_column=target_column, prediction_length=prediction_length,
                                             seasonality=seasonality)
    optional_columns = quantiles
    optional_columns = optional_columns.assign(seasonal_error=seasonal_error_rep)
    optional_columns = optional_columns.assign(item_id=forecast_item_ids)

    return result(output_file=config.output_predictions_file,
                  predictions=predictions_only,
                  truth=truth_only,
                  probabilities=None,
                  probabilities_labels=None,
                  target_is_encoded=False,
                  models_count=num_models_trained,
                  training_duration=training.duration,
                  predict_duration=predict.duration,
                  optional_columns=optional_columns)

def load_data(train_path, test_path, timestamp_column, id_column):

    train_df = pd.read_csv(
        train_path,
        parse_dates=[timestamp_column],
    )

    train_data = TimeSeriesDataFrame.from_data_frame(
        train_df,
        id_column=id_column,
        timestamp_column=timestamp_column,
    )

    test_df = pd.read_csv(
        test_path,
        parse_dates=[timestamp_column],
    )

    test_data = TimeSeriesDataFrame.from_data_frame(
        test_df,
        id_column=id_column,
        timestamp_column=timestamp_column,
    )

    return train_data, test_data


def get_eval_metric(config):
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


def save_artifacts(predictor, leaderboard, config, test_data, predictions):
    artifacts = config.framework_params.get('_save_artifacts', ['leaderboard', 'metrics'])
    try:
        if 'leaderboard' in artifacts:
            leaderboard_dir = output_subdir("leaderboard", config)
            save_pd.save(path=os.path.join(leaderboard_dir, "leaderboard.csv"), df=leaderboard)

        if 'metrics' in artifacts:
            metrics_dir = output_subdir('metrics', config)
            metrics_keys = ['MSE', 'RMSE', 'MAPE', 'MASE', 'sMAPE', 'mean_wQuantileLoss']
            metrics = {}
            for key in metrics_keys:
                #metrics[key] = [predictor.evaluate(data=test_data, metric='MSE')]
                evaluator = TimeSeriesEvaluator(
                    eval_metric=key,
                    prediction_length=predictor.prediction_length,
                    target_column=predictor.target,
                )
                metrics[key] = [evaluator(test_data, predictions)]
            metrics_df = pd.DataFrame(metrics)
            save_pd.save(path=os.path.join(metrics_dir, 'metrics.csv'), df=metrics_df)

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

def calc_seasonal_error(dataset_test, id_column, target_column, prediction_length, seasonality):
    """Calculates the sesonal error for the test dataset and repeates it for each element in the forecast sequence.

    Args:
        dataset_test (pd.DataFrame) : Dataframe containing target and item id column, shape (N, K>=2)
        id_column (str) : Name of item id column.
        target_column (str) : Name of target column.
        prediction_length (int) : Prediction length which is evaluated.
    Returns:
        seasonal_error_rep (np.ndarray) : Naive 1 error for each sequence. Shape (N,)

    """

    dtype=dataset_test[target_column].dtype
    # we aim to calculate the mean period error from the past for each sequence: 1/N sum_{i=1}^N |x(t_i) - x(t_i - T)|
    # 1. retrieve item_ids for each sequence/item
    #dataset..X /. y
    unique_item_ids, unique_item_ids_indices, unique_item_ids_inverse = np.unique(dataset_test.reset_index()[id_column].squeeze().to_numpy(), return_index=True, return_inverse=True)

    # 2. capture sequences in a list
    y_past = [dataset_test[target_column].squeeze().to_numpy(dtype=dtype)[unique_item_ids_inverse == i][:-prediction_length] for i in np.argsort(unique_item_ids_indices)]
    # 3. calculate period error per sequence
    seasonal_error = np.array([np.mean(np.abs(y_past_item[seasonality:] - y_past_item[:-seasonality])) for y_past_item in y_past], dtype=dtype)
    # 4. repeat period error for each sequence, to save one for each element
    seasonal_error_rep = np.repeat(seasonal_error, prediction_length)

    return seasonal_error_rep

def get_seasonality_from_freq(freq: str) -> int:
    """Calculates the seasonality from a frequency.

    Args:
        freq (str) : Frequency of time series.
    Returns:
        seasonality (int) : Seasonality.

    >>> get_seasonality_from_freq("2H")
    12
    """

    seasonalities = {
        "S": 3600,  # 1 hour
        "T": 1440,  # 1 day
        "H": 24,  # 1 day
        "D": 1,  # 1 day
        "W": 1,  # 1 week
        "M": 12, # 1 year
        "B": 5,  # 1 business week
        "Q": 4,  # 1 year
    }

    offset = pd.tseries.frequencies.to_offset(freq)
    base_seasonality = seasonalities.get(offset.name.split("-")[0], 1)
    seasonality, remainder = divmod(base_seasonality, offset.n)
    if not remainder:
        return seasonality

    log.warning(f'Multiple {offset.n} does not divide base seasonality {base_seasonality}.')
    log.warning(f'Falling back to seasonality 1.')
    return 1

if __name__ == '__main__':
    call_run(run)
