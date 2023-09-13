import logging
import numpy as np
import pandas as pd
import warnings

warnings.simplefilter('ignore')

from neuralprophet import NeuralProphet

from frameworks.shared.callee import call_run, result
from frameworks.shared.utils import Timer, load_timeseries_dataset

log = logging.getLogger(__name__)


def run(dataset, config):
    train_df, test_df = load_timeseries_dataset(dataset)
    train_data = rename_columns(train_df, dataset)

    model = NeuralProphet(quantiles=config.quantile_levels)
    # Suppress info messages
    np_logger = logging.getLogger("NP.df_utils")
    np_logger.setLevel(logging.ERROR)

    with Timer() as training:
        model.fit(train_data, freq=dataset.freq, progress=False)

    test_data = rename_columns(test_df, dataset)
    truth_only = test_data['y'].values.copy()
    # Hide target values before forecast
    test_data['y'] = np.nan

    with Timer() as predict:
        predictions = model.predict(test_data)

    predictions_only = predictions["yhat1"].values

    optional_columns = dict(
        repeated_item_id=np.load(dataset.repeated_item_id),
        repeated_abs_seasonal_error=np.load(dataset.repeated_abs_seasonal_error),
    )
    for q in config.quantile_levels:
        if str(q) == "0.5":
            col_name = "yhat1"
        else:
            col_name = f"yhat1 {q:.1%}"
        optional_columns[str(q)] = predictions[col_name].values

    # Sanity check - make sure predictions are ordered correctly
    if (predictions['ID'] != test_data['ID']).any():
        raise AssertionError(
            "item_id column for predictions doesn't match test data index"
        )

    return result(
        output_file=config.output_predictions_file,
        predictions=predictions_only,
        truth=truth_only,
        target_is_encoded=False,
        models_count=1,
        training_duration=training.duration,
        predict_duration=predict.duration,
        optional_columns=pd.DataFrame(optional_columns),
    )


def rename_columns(df, dataset):
    return df.rename(columns={dataset.id_column: 'ID', dataset.timestamp_column: 'ds', dataset.target: 'y'})


if __name__ == '__main__':
    call_run(run)
