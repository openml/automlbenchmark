from automl.utils import call_script_in_same_dir, dir_of, pip_install


def setup():
    call_script_in_same_dir(__file__, "setup.sh")
    # pip_install('https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt', True)
    # pip_install('automl/frameworks/autosklearn/py_requirements.txt', True)


def run(*args, **kwargs):
    from .exec import run
    run(*args, **kwargs)


def docker_commands():
    return """
RUN {here}/setup.sh
""".format(here=dir_of(__file__, True))


__all__ = (setup, run, docker_commands)
