from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.resources import config as rconfig
from amlb.utils import call_script_in_same_dir


def setup(*args, **kwargs):
    call_script_in_same_dir(__file__, "setup.sh", rconfig().root_dir, *args, **kwargs)


def run(dataset: Dataset, config: TaskConfig):
    from frameworks.shared.caller import run_in_venv

    data = dict(
        train_path=dataset.train.path,
        test_path=dataset.test.path,
        target=dataset.target.name
    )

    return run_in_venv(__file__, "exec.py",
                       input_data=data, dataset=dataset, config=config)

