
from amlb.utils import call_script_in_same_dir
from amlb.benchmark import TaskConfig
from amlb.data import Dataset, DatasetType
from copy import deepcopy


def setup(*args, **kwargs):
    call_script_in_same_dir(__file__, "setup.sh", *args, **kwargs)

def run(dataset: Dataset, config: TaskConfig):
    from frameworks.shared.caller import run_in_venv

    if dataset.type is not DatasetType.timeseries:

        data = dict(
            train=dict(path=dataset.train.data_path('parquet')),
            test=dict(path=dataset.test.data_path('parquet')),
            target=dict(
                name=dataset.target.name,
                classes=dataset.target.values
            ),
            problem_type=dataset.type.name  # AutoGluon problem_type is using same names as amlb.data.DatasetType
        )
        exec_file = "exec.py"

    else:
        dataset = deepcopy(dataset)
        if not hasattr(dataset, 'timestamp_column'):
            dataset.timestamp_column = None
        if not hasattr(dataset, 'id_column'):
            dataset.id_column = None
        if not hasattr(dataset, 'prediction_length'):
            raise AttributeError("Unspecified `prediction_length`.")

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
            prediction_length=dataset.prediction_length
        )
        exec_file = "exec_ts.py"

    return run_in_venv(__file__, exec_file,
                       input_data=data, dataset=dataset, config=config)
