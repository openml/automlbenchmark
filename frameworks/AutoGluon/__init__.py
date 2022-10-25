
from amlb.utils import call_script_in_same_dir
from amlb.benchmark import TaskConfig
from amlb.data import Dataset, DatasetType
from copy import deepcopy
import os

def setup(*args, **kwargs):
    call_script_in_same_dir(__file__, "setup.sh", *args, **kwargs)

def run(dataset: Dataset, config: TaskConfig):
    env_vars = read_setup_env_vars()

    if 'MODULE' not in env_vars or not env_vars['MODULE'] == "timeseries":
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
    dataset = deepcopy(dataset)
    if not hasattr(dataset, 'timestamp_column'):
        dataset.timestamp_column = None
    if not hasattr(dataset, 'id_column'):
        dataset.id_column = None
    if not hasattr(dataset, 'forecast_horizon_in_steps'):
        raise AttributeError("Unspecified `forecast_horizon_in_steps`.")

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
        forecast_horizon_in_steps=dataset.forecast_horizon_in_steps
    )

    return run_in_venv(__file__, "exec_ts.py",
                       input_data=data, dataset=dataset, config=config)

def read_setup_env_vars():
    env_vars = {}
    fpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), '.setup', 'setup_env')
    try:
        with open(fpath, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                key, value = line.strip().split('=', 1)
                env_vars[key] = value
    except OSError:
        print(f'Could not open/read file {fpath}')
    return env_vars
