from amlb.utils import call_script_in_same_dir
from amlb.benchmark import TaskConfig
from amlb.data import Dataset, DatasetType


def setup(*args, **kwargs):
    call_script_in_same_dir(__file__, "setup.sh", *args, **kwargs)


def run(dataset: Dataset, config: TaskConfig):
    from frameworks.shared.caller import run_in_venv

    if dataset.type != DatasetType.timeseries:
        raise AssertionError(f"Framework StatsForecast only supports dataset type 'timeseries' but received '{dataset.type}'")

    data = dict(
        train_path=dataset.train.path,
        test_path=dataset.test.path,
        target=dataset.target.name,
        id_column=dataset.id_column,
        timestamp_column=dataset.timestamp_column,
        forecast_horizon_in_steps=dataset.forecast_horizon_in_steps,
        freq=dataset.freq,
        seasonality=dataset.seasonality,
        repeated_abs_seasonal_error=dataset.repeated_abs_seasonal_error,
        repeated_item_id=dataset.repeated_item_id,
    )

    return run_in_venv(__file__, "exec.py", input_data=data, dataset=dataset, config=config)
