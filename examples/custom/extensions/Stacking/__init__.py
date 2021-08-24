from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.utils import call_script_in_same_dir


def setup(*args, **kwargs):
    call_script_in_same_dir(__file__, "setup.sh", *args, **kwargs)


def run(dataset: Dataset, config: TaskConfig):
    from amlb.datautils import impute_array
    from frameworks.shared.caller import run_in_venv

    X_train_enc, X_test_enc = impute_array(dataset.train.X_enc, dataset.test.X_enc)
    data = dict(
        train=dict(
            X_enc=X_train_enc,
            y_enc=dataset.train.y_enc
        ),
        test=dict(
            X_enc=X_test_enc,
            y_enc=dataset.test.y_enc
        )
    )

    return run_in_venv(__file__, "exec.py",
                       input_data=data, dataset=dataset, config=config)

