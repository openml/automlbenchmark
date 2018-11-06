from automl.utils import call_script_in_same_dir


def setup(verbose=False):
    #call_script_in_same_dir(__file__, "setup.sh", verbose)
    pass

def run(*args, **kwargs):
    from .exec import run
    run(*args, **kwargs)


def docker_commands():
    return """
RUN apt-get install -y openjdk-8-jdk
RUN $PIP install --no-cache-dir -r automl/frameworks/H2OAutoML/py_requirements.txt
RUN $PIP install http://h2o-release.s3.amazonaws.com/h2o/master/4402/Python/h2o-3.21.0.4402-py2.py3-none-any.whl

EXPOSE 54321
EXPOSE 54322
"""


__all__ = (setup, run, docker_commands)
