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
        train_aux=dict(path=dataset.train_auxilary_data),
        test_aux=dict(path=dataset.test_auxilary_data),
        target=dict(
            name=dataset.target.name,
            classes=dataset.target.values
        ),
        problem_type=dataset.type.name  # AutoGluon problem_type is using same names as amlb.data.DatasetType
    )

    return run_in_venv(__file__, "exec.py",
                       input_data=data, dataset=dataset, config=config)

