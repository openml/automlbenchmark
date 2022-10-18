import logging
import os
import warnings
import sys
import numpy as np
from gluonts.evaluation import Evaluator
warnings.simplefilter("ignore")

if sys.platform == 'darwin':
    os.environ['OMP_NUM_THREADS'] = '1'

import pandas as pd

import time

from gluonts import __version__
from gluonts.dataset.pandas import PandasDataset
from gluonts.mx import Trainer
from gluonts.mx.trainer.callback import Callback
from gluonts.evaluation import make_evaluation_predictions


from gluonts.model.npts import NPTSEstimator
from gluonts.model.prophet import ProphetPredictor
from gluonts.model.seasonal_naive import SeasonalNaivePredictor
from gluonts.model.r_forecast import RForecastPredictor

from gluonts.model.deepar import DeepAREstimator
from gluonts.model.n_beats import NBEATSEstimator
from gluonts.model.seq2seq import MQCNNEstimator, MQRNNEstimator
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.model.tft import TemporalFusionTransformerEstimator


from frameworks.shared.callee import call_run, result, output_subdir
from frameworks.shared.utils import Timer, zip_path
import csv

log = logging.getLogger(__name__)


def run(dataset, config):
    log.info(f"\n**** GluonTS [v{__version__}] ****\n")

    timestamp_column = dataset.timestamp_column
    id_column = dataset.id_column
    prediction_length = dataset.forecast_range_in_steps
    target_column = dataset.target.name
    time_limit = config.max_runtime_seconds
    model=config.framework_params["model"]
    item_ids =  dataset.train.X[id_column].unique()
    items_indices_timestamp = [dataset.train.X[dataset.train.X[id_column] == item_id].set_index(timestamp_column).index for item_id in item_ids[:100]]
    items_freqs = [item_id_indices_timestamp.freq or item_id_indices_timestamp.inferred_freq for item_id_indices_timestamp in items_indices_timestamp]
    items_freqs_unique = set(items_freqs)
    if not len(items_freqs_unique) == 1:
        msg=f"Found not exactly one frequency across all items. Unique inferred frequencies are {items_freqs_unique}"
        raise ValueError(msg)
    freq = items_freqs[0]

    dataset_train = pd.concat([dataset.train.X, dataset.train.y], axis=1)
    dataset_test = pd.concat([dataset.test.X, dataset.test.y], axis=1)
    test_data_future = pd.concat([item_id_and_df[1].iloc[-prediction_length:] for item_id_and_df in dataset_test.groupby(id_column)], axis=0)

    dataset_train_gluonts = PandasDataset.from_long_dataframe(dataframe=dataset_train, item_id=id_column, target=target_column, timestamp=timestamp_column, freq=freq)
    dataset_test_gluonts = PandasDataset.from_long_dataframe(dataframe=dataset_test, item_id=id_column, target=target_column, timestamp=timestamp_column, freq=freq)

    with Timer() as training:
        predictor = estimate_predictor(model, prediction_length, freq, dataset_train_gluonts, time_limit)

    with Timer() as predict:
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=dataset_test_gluonts,  # test dataset
            predictor=predictor,  # predictor
            num_samples=100,  # number of sample paths we want for evaluation
        )

    quantiles_steps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    forecasts = list(forecast_it)
    tss = list(ts_it)
    quantiles = np.array([[forecast.quantile(quantile_step) for forecast in forecasts] for quantile_step in quantiles_steps], dtype=test_data_future[target_column])
    quantiles = pd.DataFrame(quantiles.reshape(9, -1).T, columns=[str(quantile_step) for quantile_step in quantiles_steps])

    predictions_only = quantiles['0.5'].values
    truth_only = test_data_future[target_column].values

    evaluator = Evaluator(quantiles=quantiles_steps)
    agg_metrics, item_metrics = evaluator(tss, forecasts)
    save_artifacts(agg_metrics, item_metrics, config)

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

    optional_columns = quantiles
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

def estimate_predictor(model, prediction_length, freq, dataset_train_gluonts, time_limit):
    if model == "Prophet":
        predictor = ProphetPredictor(prediction_length=prediction_length)

    elif model == "DeepAR":
        estimator = DeepAREstimator(prediction_length=prediction_length, freq=freq, trainer=Trainer(callbacks = [TimeLimitCallback(time_limit)]))
        predictor = estimator.train(dataset_train_gluonts)

    elif model == "NBEATS":
        estimator = NBEATSEstimator(prediction_length=prediction_length, freq=freq, trainer=Trainer(callbacks = [TimeLimitCallback(time_limit)]))
        predictor = estimator.train(dataset_train_gluonts)

    elif model == "NPTS":
        estimator = NPTSEstimator(prediction_length=prediction_length, freq=freq)
        predictor = estimator.train(dataset_train_gluonts)

    elif model == "SeasonalNaive":
        predictor = SeasonalNaivePredictor(prediction_length=prediction_length, freq=freq)

    elif model == "MQCNN":
        estimator = MQCNNEstimator(prediction_length=prediction_length, freq=freq, trainer=Trainer(callbacks = [TimeLimitCallback(time_limit)]))
        predictor = estimator.train(dataset_train_gluonts)

    elif model == "MQRNN":
        estimator = MQRNNEstimator(prediction_length=prediction_length, freq=freq, trainer=Trainer(callbacks = [TimeLimitCallback(time_limit)]))
        predictor = estimator.train(dataset_train_gluonts)

    elif model == "SimpleFeedForward":
        estimator = SimpleFeedForwardEstimator(prediction_length=prediction_length, trainer=Trainer(callbacks = [TimeLimitCallback(time_limit)]))
        predictor = estimator.train(dataset_train_gluonts)

    elif model == "TFT":
        estimator = TemporalFusionTransformerEstimator(prediction_length=prediction_length, freq=freq, trainer=Trainer(callbacks = [TimeLimitCallback(time_limit)]))
        predictor = estimator.train(dataset_train_gluonts)

    elif model == "ARIMA":
        predictor = RForecastPredictor(prediction_length=prediction_length, freq=freq, method_name="arima")

    elif model == "ETS":
        predictor = RForecastPredictor(prediction_length=prediction_length, freq=freq, method_name="ets")

    elif model == "STL-AR":
        predictor = RForecastPredictor(prediction_length=prediction_length, freq=freq, method_name="stlar")

    elif model == "Theta":
        predictor = RForecastPredictor(prediction_length=prediction_length, freq=freq, method_name="thetaf")

    else:
        msg = f"Not implemented model {model}."
        raise ValueError(msg)
    return predictor

class TimeLimitCallback(Callback):
    """GluonTS callback object to terminate training early if autogluon time limit
    is reached."""

    def __init__(self, time_limit=None):
        self.start_time = None
        self.time_limit = time_limit

    def on_train_start(self, **kwargs) -> None:
        self.start_time = time.time()

    def on_epoch_end(
        self,
        **kwargs,
    ) -> bool:
        if self.time_limit is not None:
            cur_time = time.time()
            if cur_time - self.start_time > self.time_limit:
                log.warning("Time limit exceed during training, stop training.")
                return False
        return True

def save_artifacts(agg_metrics, item_metrics, config):
    artifacts = config.framework_params.get('_save_artifacts', ['agg_metrics'])
    try:
        metrics_dir = output_subdir('metrics', config)
        if 'agg_metrics' in artifacts:
            with open(os.path.join(metrics_dir, 'agg_metrics.csv'), 'w') as f: # b
                w = csv.writer(f)
                w.writerow(agg_metrics.keys())
                w.writerow(agg_metrics.values())
        if 'item_metrics' in artifacts:
            item_metrics.to_csv(os.path.join(metrics_dir, 'item_metrics.csv'))
    except Exception:
        log.warning("Error when saving artifacts.", exc_info=True)


if __name__ == '__main__':
    call_run(run)
