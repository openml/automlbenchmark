import logging
import numpy as np
import pandas as pd
import warnings

warnings.simplefilter('ignore')

from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA,
    AutoETS,
    AutoTheta,
    Naive,
    SeasonalNaive,
)

from frameworks.shared.callee import call_run, result
from frameworks.shared.utils import Timer, load_timeseries_dataset

log = logging.getLogger(__name__)


def run(dataset, config):
    np.random.seed(config.seed)
    train_df, test_df = load_timeseries_dataset(dataset)

    print(f"debug, dataset.forecast_horizon_in_steps {dataset.forecast_horizon_in_steps}")
    print(f"debug, train_df {train_df.shape}, test_df {test_df.shape}")

    train_data = train_df.rename(
        columns={
            dataset.id_column: 'unique_id',
            dataset.timestamp_column: 'ds',
            dataset.target: 'y',
        },
    )

    models = get_models(
        framework_params=config.framework_params,
        seasonality=7 if dataset.freq == "D" else dataset.seasonality,
    )
    model_names = [repr(m) for m in models]
    # Convert quantile_levels (floats in (0, 1)) to confidence levels (ints in [0, 100]) used by StatsForecast
    levels = []
    for q in config.quantile_levels:
        level = round(abs(q - 0.5) * 200)
        levels.append(level)
    levels = sorted(list(set(levels)))

    sf = StatsForecast(
        models=models,
        freq=dataset.freq,
        n_jobs=config.cores,
        fallback_model=SeasonalNaive(season_length=dataset.seasonality),
    )

    print(f"debug, train_data shape {train_data.shape}, levels {levels}")
    with Timer() as predict:
        predictions = sf.forecast(
            df=train_data, h=dataset.forecast_horizon_in_steps, level=levels
        )

    print(f"debug, predictions {predictions.shape}, horizon {dataset.forecast_horizon_in_steps}")

    optional_columns = dict(
        repeated_item_id=np.load(dataset.repeated_item_id),
        repeated_abs_seasonal_error=np.load(dataset.repeated_abs_seasonal_error),
    )
    for q in config.quantile_levels:
        suffix = quantile_to_suffix(q)
        optional_columns[str(q)] = predictions[[m + suffix for m in model_names]].median(axis=1).values

    predictions_only = predictions[model_names].median(axis=1).values
    truth_only = test_df[dataset.target].values

    print(f"debug, predictions median {predictions_only.shape}")
    print(f"truth, truth {truth_only.shape}")
    print(f"debug, truth type {type(truth_only)}")
    print(f"debug, predictions_only head {predictions_only[:5]}")

    # Sanity check - make sure predictions are ordered correctly
    if (predictions.index != test_df[dataset.id_column]).any():
        raise AssertionError(
            "item_id column for predictions doesn't match test data index"
        )

    return result(
        output_file=config.output_predictions_file,
        predictions=predictions_only,
        truth=truth_only,
        target_is_encoded=False,
        models_count=len(models),
        training_duration=0.0,
        predict_duration=predict.duration,
        optional_columns=pd.DataFrame(optional_columns),
    )


def quantile_to_suffix(q: float) -> str:
    if q < 0.5:
        prefix = "-lo-"
        level = 100 - 200 * q
    else:
        prefix = "-hi-"
        level = 200 * q - 100
    return prefix + str(int(level))


def get_models(framework_params: dict, seasonality: int):
    model_name = framework_params.get('model_name', 'SeasonalNaive').lower()
    extra_params = {
        k: v
        for k, v in framework_params.items()
        if not (k.startswith('_') or k == 'model_name')
    }
    if model_name == 'naive':
        return [Naive()]
    elif model_name == 'seasonalnaive':
        return [SeasonalNaive(season_length=seasonality)]
    elif model_name == 'autoarima':
        return [AutoARIMA(season_length=seasonality, **extra_params)]
    elif model_name == 'autoets':
        return [AutoETS(season_length=seasonality, **extra_params)]
    elif model_name == 'autotheta':
        return [AutoTheta(season_length=seasonality, **extra_params)]
    elif model_name == 'statensemble':
        return [
            AutoARIMA(season_length=seasonality),
            AutoETS(season_length=seasonality),
            AutoTheta(season_length=seasonality),
        ]
    else:
        raise ValueError(f'Unsupported model name {model_name}')


if __name__ == '__main__':
    call_run(run)
