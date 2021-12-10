from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.utils import call_script_in_same_dir


def setup(*args, **kwargs):
    call_script_in_same_dir(__file__, "setup.sh", *args, **kwargs)


def run(dataset: Dataset, config: TaskConfig):
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
    if dataset.train.has_auxiliary_data:
        data['train_auxiliary_data'] = dict(path=dataset.train.auxiliary_data.path)
    if dataset.test.has_auxiliary_data:
        data['test_auxiliary_data'] = dict(path=dataset.test.auxiliary_data.path) 

    return run_in_venv(__file__, "exec.py",
                       input_data=data, dataset=dataset, config=config)

