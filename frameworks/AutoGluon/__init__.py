
from amlb.utils import call_script_in_same_dir
from amlb.benchmark import TaskConfig
from amlb.data import Dataset, DatasetType
from copy import deepcopy


def setup(*args, **kwargs):
    call_script_in_same_dir(__file__, "setup.sh", *args, **kwargs)

def run(dataset: Dataset, config: TaskConfig):

    if dataset.type == DatasetType.timeseries:
        return run_autogluon_timeseries(dataset, config)
    else:
        return run_autogluon_tabular(dataset, config)


def run_autogluon_tabular(dataset: Dataset, config: TaskConfig):
    from frameworks.shared.caller import run_in_venv
    data = dict(
        train=dict(path=dataset.train.data_path('parquet')),
        test=dict(path=dataset.test.data_path('parquet')),
        target=dict(
            name=dataset.target.name,
            classes=dataset.target.values
        ),
        problem_type=dataset.type.name,  # AutoGluon problem_type is using same names as amlb.data.DatasetType
    )
    if config.measure_inference_time:
        data["inference_subsample_files"] = dataset.inference_subsample_files(fmt="parquet")

    return run_in_venv(__file__, "exec.py",
                       input_data=data, dataset=dataset, config=config)

def run_autogluon_timeseries(dataset: Dataset, config: TaskConfig):
    from frameworks.shared.caller import run_in_venv
    dataset = deepcopy(dataset)

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

    return run_in_venv(__file__, "exec_ts.py",
                       input_data=data, dataset=dataset, config=config)
