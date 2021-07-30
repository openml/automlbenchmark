from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.utils import call_script_in_same_dir, unsparsify


def setup(*args, **kwargs):
    call_script_in_same_dir(__file__, "setup.sh", *args, **kwargs)


def run(dataset: Dataset, config: TaskConfig):
    from frameworks.shared.caller import run_in_venv

    X_train, X_test = dataset.train.X_enc, dataset.test.X_enc
    y_train, y_test = unsparsify(dataset.train.y_enc, dataset.test.y_enc)
    data = dict(
        train=dict(
            X=X_train,
            y=y_train
        ),
        test=dict(
            X=X_test,
            y=y_test
        ),
        predictors_type=['Numerical' if p.is_numerical() else 'Categorical' for p in dataset.predictors]
    )

    return run_in_venv(__file__, "exec.py",
                       input_data=data, dataset=dataset, config=config)

