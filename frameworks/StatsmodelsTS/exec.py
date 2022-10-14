import logging
import os
import warnings
import sys
import numpy as np
warnings.simplefilter("ignore")

if sys.platform == 'darwin':
    os.environ['OMP_NUM_THREADS'] = '1'

import pandas as pd

import time

from statsmodels import __version__
import statsmodels as sm

from frameworks.shared.callee import call_run, result, output_subdir
from frameworks.shared.utils import Timer, zip_path

log = logging.getLogger(__name__)

import statsmodels as sm
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.api import STLForecast
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing


def run(dataset, config):
    log.info(f"\n**** StatsmodelsTS [v{__version__}] ****\n")

    timestamp_column = dataset.timestamp_column
    id_column = dataset.id_column
    prediction_length = dataset.forecast_range_in_steps
    target_column = dataset.target.name
    model=config.framework_params["model"]
    item_ids =  dataset.train.X[id_column].unique()
    items_indices_timestamp = [dataset.train.X[dataset.train.X[id_column] == item_id].set_index(timestamp_column).index for item_id in item_ids[:100]]
    items_freqs = [item_id_indices_timestamp.freq or item_id_indices_timestamp.inferred_freq for item_id_indices_timestamp in items_indices_timestamp]
    items_freqs_unique = set(items_freqs)
    if not len(items_freqs_unique) == 1:
        msg=f"Found not exactly one frequency across all items. Unique inferred frequencies are {items_freqs_unique}"
        raise ValueError(msg)
    freq = items_freqs[0]

    dataset_test = pd.concat([dataset.test.X, dataset.test.y], axis=1)
    test_data_past = pd.concat([item_id_and_df[1].iloc[:-dataset.forecast_range_in_steps] for item_id_and_df in dataset_test.groupby(dataset.id_column)], axis=0)
    test_data_future = pd.concat([item_id_and_df[1].iloc[-dataset.forecast_range_in_steps:] for item_id_and_df in dataset_test.groupby(dataset.id_column)], axis=0)

    dataset_test = pd.concat([dataset.test.X, dataset.test.y], axis=1)
    test_dataset_past_items = [item[1].set_index(dataset.timestamp_column)[dataset.target.name] for item in list(test_data_past.groupby(dataset.id_column))]
    # why from statsmodels.tsa.statespace.sarimax import SARIMAX as StatsmodelsSARIMAX instead of sm.tsa.arima.model.ARIMA ?
    # why not STL-AR ? Can we use
    # are we using sk time?
    # we only get variance, which we can use to calculate quantiles?
    # so ETS and ARIMA are sample methods and not quantile methods? should this bother us?

    with Timer() as training:
        pass

    with Timer() as predict:
        pred_mean = []
        pred_quantiles = []
        for endog in test_dataset_past_items:
            cutoff = endog.index.max()
            endog.index.freq = freq
            start = cutoff + pd.tseries.frequencies.to_offset(freq)
            end = cutoff + dataset.forecast_range_in_steps * pd.tseries.frequencies.to_offset(freq)

            model_name = config.framework_params["model"]

            quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            quantile_levels_text = [str(quantile_level) for quantile_level in quantile_levels]
            coverages = [2 * quantile_level if quantile_level <0.5 else 2 * (1 - quantile_level) for quantile_level in quantile_levels]
            cols_indices = [0 if quantile_level < 0.5 else 1 for quantile_level in quantile_levels]

            if model_name in ['ARIMA', 'STL-AR', 'STL-ARIMA', 'STL-ETS']:
                if model_name == 'ARIMA':
                    #model = ARIMA(endog)
                    model = SARIMAX(endog)
                elif model_name == 'STL-AR':
                    model = STLForecast(endog, AutoReg, model_kwargs={"lags": None})
                elif model_name == 'STL-ARIMA':
                    model = STLForecast(endog, ARIMA)
                elif model_name == 'STL-ETS':
                    model = STLForecast(endog, ETSModel)
                res = model.fit()
                fc = res.get_prediction(start, end)
                item_pred_mean = fc.predicted_mean.values
                item_pred_quantiles = []
                for i in range(len(quantile_levels)):
                    quantile = fc.conf_int(alpha=coverages[i]).iloc[:, cols_indices[i]]
                    item_pred_quantiles.append(quantile)
                item_pred_quantiles = pd.concat(item_pred_quantiles, keys=quantile_levels_text, axis=1)

            elif model_name == 'ETS':
                model = ETSModel(endog) # -> fc.predicted_mean fc.pred_int(alpha=0.5)
                res = model.fit()
                fc = res.get_prediction(start, end)
                item_pred_mean = fc.predicted_mean.values
                item_pred_quantiles = []
                for i in range(len(quantile_levels)):
                    quantile = fc.pred_int(alpha=coverages[i]).iloc[:, cols_indices[i]]
                    item_pred_quantiles.append(quantile)
                item_pred_quantiles = pd.concat(item_pred_quantiles, keys=quantile_levels_text, axis=1)

            elif model_name == 'Theta':
                model = ThetaModel(endog) # -> res.forecast(steps=dataset.forecast_range_in_steps) res.prediction_intervals(steps=dataset.forecast_range_in_steps, alpha=0.5)
                res = model.fit()
                item_pred_mean = res.forecast(steps=dataset.forecast_range_in_steps).values
                item_pred_quantiles = []
                for i in range(len(quantile_levels)):
                    quantile = res.prediction_intervals(steps=dataset.forecast_range_in_steps, alpha=coverages[i]).iloc[:, cols_indices[i]]
                    item_pred_quantiles.append(quantile)
                item_pred_quantiles = pd.concat(item_pred_quantiles, keys=quantile_levels_text, axis=1)
            else:
                msg = f'Not implemented model {model}.'
                raise ValueError(msg)
            pred_mean.append(item_pred_mean)
            pred_quantiles.append(item_pred_quantiles)

        pred_mean = np.concatenate(pred_mean)
        pred_quantiles = pd.concat(pred_quantiles, axis=0).reset_index(drop=True)
        #print(f'mean {mean}')
        #print(f'confidence interval {quantile}')

    predictions_only = pred_mean
    truth_only = test_data_future[dataset.target.name].values

    # evaluator = Evaluator(quantiles=quantiles_steps)
    # agg_metrics, item_metrics = evaluator(tss, forecasts)
    # item_metrics['seasonal_error']

    period_length = 1 # TODO: This period length could be adapted to the Dataset, but then we need to pass this information as well. As of now this works.

    # we aim to calculate the mean period error from the past for each sequence: 1/N sum_{i=1}^N |x(t_i) - x(t_i - T)|
    # 1. retrieve item_ids for each sequence/item
    #dataset..X /. y
    item_ids, inverse_item_ids = np.unique(dataset_test.reset_index()[id_column].squeeze().to_numpy(), return_index=False, return_inverse=True)
    # 2. capture sequences in a list
    y_past = [dataset_test[target_column].squeeze().to_numpy()[inverse_item_ids == i][:-prediction_length] for i in range(len(item_ids))]
    # 3. calculate period error per sequence
    y_past_period_error = [np.abs(y_past_item[period_length:] - y_past_item[:-period_length]).mean() for y_past_item in y_past]
    # 4. repeat period error for each sequence, to save one for each element
    y_past_period_error_rep = np.repeat(y_past_period_error, prediction_length)

    optional_columns = pred_quantiles
    optional_columns = optional_columns.assign(y_past_period_error=y_past_period_error_rep)

    return result(output_file=config.output_predictions_file,
                  predictions=predictions_only,
                  truth=truth_only,
                  probabilities=None,
                  probabilities_labels=None,
                  target_is_encoded=False,
                  models_count=1, #num_models_trained,
                  training_duration=training.duration,
                  predict_duration=predict.duration,
                  optional_columns=optional_columns)


if __name__ == '__main__':
    call_run(run)
