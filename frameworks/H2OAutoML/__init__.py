from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.resources import config as rconfig
from amlb.utils import call_script_in_same_dir


def setup(*args, **kwargs):
    call_script_in_same_dir(__file__, "setup.sh", *args, **kwargs)


def run(dataset: Dataset, config: TaskConfig):
    from frameworks.shared.caller import run_in_venv
    data = dict(
        train=dict(path=dataset.train.path),
        test=dict(path=dataset.test.path),
        target=dict(index=dataset.target.index),
        domains=dict(cardinalities=[0 if f.values is None else len(f.values) for f in dataset.features]),
        format=dataset.train.format,
    )
    if config.measure_inference_time:
        data["inference_subsample_files"] = dataset.inference_subsample_files(fmt=dataset.train.format, with_labels=True)
    config.ext.monitoring = rconfig().monitoring
    return run_in_venv(__file__, "exec.py",
                       input_data=data, dataset=dataset, config=config)


def docker_commands(*args, setup_cmd=None):
    return """
{cmd}
EXPOSE 54321
EXPOSE 54322
""".format(
        cmd="RUN {}".format(setup_cmd) if setup_cmd is not None else ""
    )


# There is no network isolation in Singularity,
#  so there is no need to map any port.
# If the process inside the container binds to an IP:port,
# it will be immediately reachable on the host.
def singularity_commands(*args, setup_cmd=None):
    return """
{cmd}
""".format(
        cmd="{}".format(setup_cmd) if setup_cmd is not None else ""
    )
