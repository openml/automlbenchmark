from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.resources import config as rconfig
from amlb.utils import call_script_in_same_dir, dir_of


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


def docker_commands(*args, **kwargs):
    return """
RUN {here}/setup.sh {amlb_dir}
""".format(here=dir_of(__file__, True), amlb_dir=rconfig().root_dir)


def singularity_commands(*args, **kwargs):
    return """
{here}/setup.sh {amlb_dir}
""".format(here=dir_of(__file__, True), amlb_dir=rconfig().root_dir)



__all__ = (setup, run, docker_commands())
