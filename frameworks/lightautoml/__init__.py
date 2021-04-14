from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.utils import call_script_in_same_dir


def setup(*args, **kwargs):
    call_script_in_same_dir(__file__, "setup.sh", *args, **kwargs)


def run(dataset: Dataset, config: TaskConfig):
    from frameworks.shared.caller import run_in_venv

    data = dict(
        train=dict(
            data=dataset.train.data,
            y_enc=dataset.train.y_enc
        ),
        test=dict(
            data=dataset.test.data,
            y_enc=dataset.test.y_enc
        ),
        target=dict(
            name=dataset.target.name,
            classes=dataset.target.values
        ),
        columns=[
            (f.name, ('object' if f.is_categorical(strict=False)
                      else 'int' if f.data_type == 'integer' else 'float')) for f in dataset.features],
        problem_type=dataset.type.name
    )

    return run_in_venv(__file__, "exec.py",
                       input_data=data, dataset=dataset, config=config)
