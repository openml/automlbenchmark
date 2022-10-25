
from amlb.utils import call_script_in_same_dir
from amlb.benchmark import TaskConfig
from amlb.data import Dataset, DatasetType
from copy import deepcopy

def setup(*args, **kwargs):
    call_script_in_same_dir(__file__, "setup.sh", *args, **kwargs)

def run(dataset: Dataset, config: TaskConfig):

    if dataset.type is not DatasetType.timeseries:
        raise ValueError("Framework `GluonTS` does exepct timeseries tasks.")

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
        train=dict(X=dataset.train.X, y=dataset.train.y, path=dataset.train.path),
        test=dict(X=dataset.test.X, y=dataset.test.y, path=dataset.test.path),
        target=dict(
            name=dataset.target.name,
            classes=dataset.target.values
        ),
        problem_type=dataset.type.name,  # AutoGluon problem_type is using same names as amlb.data.DatasetType
        timestamp_column=dataset.timestamp_column,
        id_column=dataset.id_column,
        forecast_horizon_in_steps=dataset.forecast_horizon_in_steps
    )

    return run_in_venv(__file__, "exec.py",
                       input_data=data, dataset=dataset, config=config)
