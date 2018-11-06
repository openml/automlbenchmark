from automl.utils import call_script_in_same_dir


def setup(verbose=True):
    # call_script_in_same_dir(__file__, "setup.sh", verbose)
    pass


def run(*args, **kwargs):
    from .exec import run
    run(*args, **kwargs)


def docker_commands():
    return """
RUN apt-get install -y build-essential swig
RUN curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip3 install
RUN $PIP install --no-cache-dir -r automl/frameworks/autosklearn/py_requirements.txt
"""


__all__ = (setup, run, docker_commands)
