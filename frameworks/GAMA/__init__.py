from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.utils import call_script_in_same_dir


def setup(*args, **kwargs):
    call_script_in_same_dir(__file__, "setup.sh", *args, **kwargs)


def run(dataset: Dataset, config: TaskConfig):
    from frameworks.shared.caller import run_in_venv

    data = dict(
        target=dataset.target.name,
        train=dict(
            path=dataset.train.path,
            X=dataset.train.X,
            y=dataset.train.y
        ),
        test=dict(
            path=dataset.test.path,
            X=dataset.test.X,
            y=dataset.test.y
        ),
    )
    options = dict(
        serialization=dict(sparse_dataframe_deserialized_format='dense')
    )

    return run_in_venv(__file__, "exec.py",
                       input_data=data, dataset=dataset, config=config, options=options)

