from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.resources import config as rconfig
from amlb.utils import call_script_in_same_dir, dir_of


def setup(*args, **kwargs):
    call_script_in_same_dir(__file__, "setup.sh", rconfig().root_dir, *args, **kwargs)


def run(dataset: Dataset, config: TaskConfig):
    from amlb.datautils import impute
    from frameworks.shared.caller import run_in_venv

    # Disable imputation/encoding later.
    X_train_enc, X_test_enc = impute(dataset.train.X_enc, dataset.test.X_enc)
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


def docker_commands(*args, **kwargs):
    return """
RUN {here}/setup.sh {amlb_dir}
""".format(here=dir_of(__file__, True), amlb_dir=rconfig().root_dir)


__all__ = (setup, run, docker_commands())
