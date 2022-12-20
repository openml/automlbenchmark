import logging
import os
import warnings
import sys
import numpy as np
warnings.simplefilter("ignore")

if sys.platform == 'darwin':
    os.environ['OMP_NUM_THREADS'] = '1'

import pandas as pd

from autoPyTorch import __version__
from autoPyTorch.api.time_series_forecasting import TimeSeriesForecastingTask
from autoPyTorch.datasets.time_series_dataset import TimeSeriesSequence

from frameworks.shared.callee import call_run, result
from frameworks.shared.utils import Timer

log = logging.getLogger(__name__)


def run(dataset, config):
    log.info(f"\n**** AutoPyTorchTS [v{__version__}] ****\n")
    log.warning("Ignoring cores constraint of %s cores.", config.cores)

    id_column = dataset.id_column
    prediction_length = dataset.forecast_horizon_in_steps
    target_column = dataset.target.name
    seasonality = dataset.seasonality

    dataset_train = pd.concat([dataset.train.X, dataset.train.y], axis=1)
    dataset_test = pd.concat([dataset.test.X, dataset.test.y], axis=1)

    forecast_horizon = dataset.forecast_horizon_in_steps
    eval_metric = get_eval_metric(config)

    y_test_future = [seq[1][dataset.target.name].reset_index(drop=True)[-forecast_horizon:] for seq in list(dataset_test.groupby(dataset.id_column))]
    y_train = [seq[1][dataset.target.name].reset_index(drop=True) for seq in list(dataset_train.groupby(dataset.id_column))]

    X_train = None
    X_test = None

    known_future_features = None

    start_times_train = [seq[1][dataset.timestamp_column].iloc[0] for seq in list(dataset.train.X.groupby(dataset.id_column))]

    log.info(f'Target name {dataset.target.name}. Forecast horizon {forecast_horizon}.')
    log.info(f'There are {len(list(dataset_test.groupby(dataset.id_column)))} sequences in the test dataset.')
    log.info(f'There are {len(list(dataset_train.groupby(dataset.id_column)))} sequences in the train dataset.')

    item_ids =  dataset.test.X[dataset.id_column].unique()
    items_indices_timestamp = [dataset.test.X[dataset.test.X[dataset.id_column] == item_id].set_index(dataset.timestamp_column).index for item_id in item_ids[:100]]
    items_freqs = [item_id_indices_timestamp.freq or item_id_indices_timestamp.inferred_freq for item_id_indices_timestamp in items_indices_timestamp]
    items_freqs_unique = set(items_freqs)
    if not len(items_freqs_unique) == 1:
        msg=f"Found not exactly one frequency across all items. Unique inferred frequencies are {items_freqs_unique}"
        raise ValueError(msg)
    freq = items_freqs[0]

    with Timer() as training:
        # initialise Auto-PyTorch api
        api = TimeSeriesForecastingTask()

        # Search for an ensemble of machine learning algorithms
        api.search(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            optimize_metric=eval_metric,
            n_prediction_steps=forecast_horizon,
            memory_limit=config.max_mem_size_mb,  # Currently, forecasting models use much more memories
            freq=freq,
            start_times=start_times_train,
            total_walltime_limit=config.max_runtime_seconds,
            min_num_test_instances=1000,  # proxy validation sets. This only works for the tasks with more than 1000 series
            known_future_features=known_future_features,
        )

    with Timer() as predict:

        test_sets = []
        for seq in list(dataset_test.groupby(dataset.id_column)):
            ts = TimeSeriesSequence(
                X=None,
                Y=seq[1][dataset.target.name].reset_index(drop=True).to_numpy()[:-forecast_horizon, None],
                start_time=seq[1][dataset.timestamp_column].iloc[0],
                freq=freq,
                time_feature_transform=api.dataset.time_feature_transform,
                train_transforms=api.dataset.train_transform,
                val_transforms=api.dataset.val_transform,
                n_prediction_steps=prediction_length,
                sp=api.dataset.seasonality,
                is_test_set=True)
            test_sets.append(ts)
        y_pred = api.predict(test_sets)

    predictions_only = np.array(y_pred, dtype=np.float64).flatten()
    truth_only = np.array(y_test_future, dtype=np.float64).flatten()

    forecast_unique_item_ids = np.arange(predictions_only.shape[0] / prediction_length)
    forecast_item_ids = np.repeat(forecast_unique_item_ids, prediction_length)

    seasonal_error_rep = calc_seasonal_error(dataset_test=dataset_test, id_column=id_column, target_column=target_column, prediction_length=prediction_length, seasonality=seasonality)

    quantiles_steps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    quantiles = pd.DataFrame(predictions_only.repeat(9).reshape(-1, 9), columns=[str(quantile_step) for quantile_step in quantiles_steps])

    log.info(f'Predictions Shape {predictions_only.shape}.')
    log.info(f'Truth Shape {truth_only.shape}.')
    log.info(f'Quantiles Shape {quantiles.shape}.')

    optional_columns = quantiles
    optional_columns = optional_columns.assign(seasonal_error=seasonal_error_rep)
    optional_columns = optional_columns.assign(item_id=forecast_item_ids)

    return result(output_file=config.output_predictions_file,
                  predictions=predictions_only,
                  truth=truth_only,
                  probabilities=None,
                  probabilities_labels=None,
                  target_is_encoded=False,
                  models_count=1,
                  training_duration=training.duration,
                  predict_duration=predict.duration,
                  optional_columns=optional_columns)


def get_eval_metric(config):
    metrics_mapping = dict(
        mase='mean_MASE_forecasting',
        median_mase='median_MASE_forecasting',
        mae='mean_MAE_forecasting',
        median_mae='median_MAE_forecasting',
        mape='mean_MAPE_forecasting',
        median_mape='median_MAPE_forecasting',
        mse='mean_MSE_forecasting',
        median_mse='median_MSE_forecasting'
    )

    eval_metric = metrics_mapping[config.metric] if config.metric in metrics_mapping else None
    if eval_metric is None:
        log.warning("Performance metric %s not supported.", config.metric)
    return eval_metric

def calc_seasonal_error(dataset_test, id_column, target_column, prediction_length, seasonality):
    """Calculates the seasonal error for the test dataset and repeates it for each element in the forecast sequence.

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


if __name__ == '__main__':
    call_run(run)
