
from amlb.utils import call_script_in_same_dir
from amlb.benchmark import TaskConfig
from amlb.data import Dataset, DatasetType
from copy import deepcopy
from copy import copy
import os

def setup(*args, **kwargs):
    call_script_in_same_dir(__file__, "setup.sh", *args, **kwargs)

def run(dataset: Dataset, config: TaskConfig):

    if 'MODULE' not in config.setup_env or not config.setup_env['MODULE'] == "timeseries":
        if dataset.type is not DatasetType.timeseries:
            return run_autogluon_tabular(dataset, config)
        else:
            msg=f'Error: Installed module autogluon.tabular but task equals DatasetType.timeseries.'
            raise ValueError(msg)

    else:
        if dataset.type is not DatasetType.timeseries:
            msg=f'Error: Installed module autogluon.timeseries but task does not equal DatasetType.timeseries.'
            raise ValueError(msg)
        else:
            return run_autogluon_timeseries(dataset, config)

def run_autogluon_tabular(dataset: Dataset, config: TaskConfig):
    from frameworks.shared.caller import run_in_venv
    data = dict(
        train=dict(path=dataset.train.data_path('parquet')),
        test=dict(path=dataset.test.data_path('parquet')),
        target=dict(
            name=dataset.target.name,
            classes=dataset.target.values
        ),
        problem_type=dataset.type.name  # AutoGluon problem_type is using same names as amlb.data.DatasetType
    )

    return run_in_venv(__file__, "exec.py",
                       input_data=data, dataset=dataset, config=config)

def run_autogluon_timeseries(dataset: Dataset, config: TaskConfig):
    from frameworks.shared.caller import run_in_venv
    dataset = copy(dataset)
    if not hasattr(dataset, 'timestamp_column'):
        dataset.timestamp_column = None
    if not hasattr(dataset, 'id_column'):
        dataset.id_column = None
    if not hasattr(dataset, 'forecast_horizon_in_steps'):
        raise AttributeError("Unspecified `forecast_horizon_in_steps`.")
    if not hasattr(dataset, 'seasonality'):
        raise AttributeError("Unspecified `seasonality`.")

    data = dict(
        # train=dict(path=dataset.train.data_path('parquet')),
        # test=dict(path=dataset.test.data_path('parquet')),
        train=dict(path=dataset.train.path),
        test=dict(path=dataset.test.path),
        target=dict(
            name=dataset.target.name,
            classes=dataset.target.values
        ),
        problem_type=dataset.type.name,  # AutoGluon problem_type is using same names as amlb.data.DatasetType
        timestamp_column=dataset.timestamp_column,
        id_column=dataset.id_column,
        forecast_horizon_in_steps=dataset.forecast_horizon_in_steps,
        seasonality=dataset.seasonality
    )

    return run_in_venv(__file__, "exec_ts.py",
                       input_data=data, dataset=dataset, config=config)
