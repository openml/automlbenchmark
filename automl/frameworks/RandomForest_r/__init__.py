from automl.utils import call_script_in_same_dir, dir_of


def setup():
    call_script_in_same_dir(__file__, "setup.sh")


def run(*args, **kwargs):
    from .exec import run
    run(*args, **kwargs)


def docker_commands():
    return """
RUN apt-get -y install software-properties-common apt-transport-https libxml2-dev
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E084DAB9
RUN add-apt-repository 'deb [arch=amd64,i386] https://cloud.r-project.org/bin/linux/ubuntu xenial-cran35/'
RUN apt-get -y install r-base r-base-dev
RUN {here}/setup.sh
""".format(here=dir_of(__file__, True))


__all__ = (setup, run, docker_commands)
