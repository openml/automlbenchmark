import logging
from importlib.metadata import version
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from frameworks.shared.callee import call_run, result
from frameworks.shared.utils import load_timeseries_dataset, Timer

from tabpfn import TabPFNRegressor
from tabpfn.model.bar_distribution import FullSupportBarDistribution

TABPFN_VERSION = version("tabpfn")
TABPFN_DEFAULT_QUANTILE = [i / 10 for i in range(1, 10)]


logger = logging.getLogger(__name__)


def split_time_series_to_X_y(df, target_col="target"):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def run(dataset, config):
    logger.info(f"**** TabPFN TimeSeries [v{TABPFN_VERSION}] ****")

    # Initialize TabPFN model
    model = TabPFNRegressor(
        device="cuda" if torch.cuda.is_available() else "cpu",
        show_progress=False,
    )
    logger.info(f"Using device: {model.device}")

    train_df, test_df = load_timeseries_dataset(dataset)

    # # Debug
    # selected_item_ids = train_df['item_id'].unique()[:3]
    # train_df = train_df[train_df['item_id'].isin(selected_item_ids)]
    # test_df = test_df[test_df['item_id'].isin(selected_item_ids)]

    # Sort and group by item_id
    group_key = "item_id"
    train_df.sort_values(by=[group_key], inplace=True)
    test_df.sort_values(by=[group_key], inplace=True)
    train_grouped = train_df.groupby(group_key)
    test_grouped = test_df.groupby(group_key)
    assert len(train_grouped) == len(test_grouped)

    # Perform prediction for each time-series
    all_pred = {"mean": []} | {str(q): [] for q in config.quantile_levels}
    predict_duration = 0.0
    for (train_id, train_group), (test_id, test_group) in tqdm(zip(train_grouped, test_grouped),
                                                               total=len(train_grouped),
                                                               desc="Processing Groups"):
        assert train_id == test_id

        train_group.drop(columns=[group_key], inplace=True)
        test_group.drop(columns=[group_key], inplace=True)

        train_X, train_y = split_time_series_to_X_y(train_group)
        test_X, test_y = split_time_series_to_X_y(test_group)

        # TabPFN fit and predict at the same time (single forward pass)
        model.fit(train_X, train_y)
        with Timer() as predict:
            pred = model.predict_full(test_X)
        predict_duration += predict.duration
        all_pred["mean"].append(pred["mean"])

        # Get quantile predictions
        for q in config.quantile_levels:
            if q in TABPFN_DEFAULT_QUANTILE:
                quantile_pred = pred[f"quantile_{q:.2f}"]   # (n_horizon, )

            else:
                criterion: FullSupportBarDistribution = pred["criterion"]
                logits = torch.tensor(pred["logits"])
                quantile_pred = criterion.icdf(logits, q).numpy()   # (n_horizon, )

            all_pred[str(q)].append(quantile_pred)

    # Concatenate all quantile predictions
    for k in all_pred.keys():
        all_pred[k] = np.concatenate(all_pred[k], axis=0)   # (n_item * n_horizon,)

    # Crucial for the result to be interpreted as TimeSeriesResults
    optional_columns = dict(
        repeated_item_id=np.load(dataset.repeated_item_id)[:test_df.shape[0]],
        repeated_abs_seasonal_error=np.load(dataset.repeated_abs_seasonal_error)[:test_df.shape[0]]
    )

    for q in config.quantile_levels:
        optional_columns[str(q)] = all_pred[str(q)]

    return result(
        output_file=config.output_predictions_file,
        predictions=all_pred["mean"],
        truth=test_df[dataset.target].values,
        target_is_encoded=False,
        models_count=1,
        training_duration=0.0,
        predict_duration=predict_duration,
        optional_columns=pd.DataFrame(optional_columns),
    )


if __name__ == '__main__':
    call_run(run)