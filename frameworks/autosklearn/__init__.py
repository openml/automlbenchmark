from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.resources import config as rconfig
from amlb.utils import call_script_in_same_dir


def setup(*args, **kwargs):
    call_script_in_same_dir(__file__, "setup.sh", rconfig().root_dir, *args, **kwargs)


def run(dataset: Dataset, config: TaskConfig):
    from frameworks.shared.caller import run_in_venv

    data = dict(
        train=dict(
            X_enc=dataset.train.X_enc,
            y_enc=dataset.train.y_enc
        ),
        test=dict(
            X_enc=dataset.test.X_enc,
            y_enc=dataset.test.y_enc
        ),
        predictors_type=['Numerical' if p.is_numerical() else 'Categorical' for p in dataset.predictors]
    )

    return run_in_venv(__file__, "exec.py",
                       input_data=data, dataset=dataset, config=config)

