from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.resources import config as rconfig
from amlb.utils import call_script_in_same_dir


def setup(*args, **kwargs):
    call_script_in_same_dir(__file__, "setup.sh", rconfig().root_dir, *args, **kwargs)


def run(dataset: Dataset, config: TaskConfig):
    from frameworks.shared.caller import run_in_venv

    data = dict(
        train=dict(data=dataset.train.data),
        test=dict(data=dataset.test.data),
        target=dict(
            name=dataset.target.name,
            classes=dataset.target.values
        ),
        columns=[(f.name, ('object' if f.is_categorical(strict=False)  # keep as object everything that is not numerical
                           else 'int' if f.data_type == 'integer'
                           else 'float')) for f in dataset.features],
        problem_type=dataset.type.name  # AutoGluon problem_type is using same names as amlb.data.DatasetType
    )

    return run_in_venv(__file__, "exec.py",
                       input_data=data, dataset=dataset, config=config)

