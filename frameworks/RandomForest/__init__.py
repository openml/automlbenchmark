from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.resources import config as rconfig
from amlb.utils import Namespace as ns, call_script_in_same_dir


def setup(*args, **kwargs):
    call_script_in_same_dir(__file__, "setup.sh", rconfig().root_dir, *args, **kwargs)


def run(dataset: Dataset, config: TaskConfig):
    from amlb.datautils import impute
    from frameworks.shared.caller import run_python_script_in_same_module

    X_train_enc, X_test_enc = impute(dataset.train.X_enc, dataset.test.X_enc)
    data = ns(
        train=ns(
            X_enc=X_train_enc,
            y_enc=dataset.train.y_enc
        ),
        test=ns(
            X_enc=X_test_enc,
            y_enc=dataset.test.y_enc
        )
    )

    return run_python_script_in_same_module(__file__, "exec.py",
                                            input_data=data, dataset=dataset, config=config)


__all__ = (run)
