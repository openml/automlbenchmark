"""
**docker** module is build on top of **ContainerBenchmark** module to provide logic to create and run docker images
that are preconfigured with a given automl framework, and that can be used to run a benchmark anywhere.
The docker image embeds a version of the automlbenchmark app so that tasks are later run in local mode inside docker,
providing the same parameters and features allowing to import config and export results through mounted folders.
"""
import logging
import os
import re

from ..benchmark import _setup_dir_
from ..resources import config as rconfig
from ..utils import dir_of, run_cmd, str_digest, str_sanitize, touch
from .container import ContainerBenchmark


log = logging.getLogger(__name__)


class DockerBenchmark(ContainerBenchmark):
    """DockerBenchmark
    an extension of ContainerBenchmark to run benchmarks inside docker.
    """

    def __init__(self, framework_name, benchmark_name, constraint_name):
        """

        :param framework_name:
        :param benchmark_name:
        :param constraint_name:
        """
        super().__init__(framework_name, benchmark_name, constraint_name)
        self._custom_image_name = rconfig().docker.image
        self.minimize_instances = rconfig().docker.minimize_instances
        self.container_name = 'docker'
        self.force_branch = rconfig().docker.force_branch
        self.custom_commands = self.framework_module.docker_commands(
            self.framework_def.setup_args,
            setup_cmd=self.framework_def._setup_cmd
        ) if hasattr(self.framework_module, 'docker_commands') else ""

    @property
    def _script(self):
        return os.path.join(self._framework_dir, _setup_dir_, 'Dockerfile')

    def _start_container(self, script_params=""):
        """Implementes the container run method"""
        in_dir = rconfig().input_dir
        out_dir = rconfig().output_dir
        custom_dir = rconfig().user_dir
        for d in [in_dir, out_dir, custom_dir]:
            touch(d, as_dir=True)

        run_as = resolve_docker_run_as_option(rconfig().docker.run_as)
        script_extra_params = "--session="  # in combination with `self.output_dirs.session` usage below to prevent creation of 2 sessions locally
        inst_name = f"{self.sid}.{str_sanitize(str_digest(script_params))}"
        cmd = (
            "docker run --name {name} {options} {run_as} "
            "-v {input}:/input -v {output}:/output -v {custom}:/custom "
            "--rm {image} {params} -i /input -o /output -u /custom -s skip -Xrun_mode=docker {extra_params}"
        ).format(
            name=inst_name,
            options=rconfig().docker.run_extra_options,
            run_as=run_as,
            input=in_dir,
            output=self.output_dirs.session,
            custom=custom_dir,
            image=self.image,
            params=script_params,
            extra_params=script_extra_params,
        )
        log.info("Starting docker: %s.", cmd)
        log.info("Datasets are loaded by default from folder %s.", in_dir)
        log.info("Generated files will be available in folder %s.", out_dir)
        try:
            run_cmd(cmd, _capture_error_=False)  # console logs are written on stderr by default: not capturing allows live display
        except:  # also want to handle KeyboardInterrupt
            try:
                run_cmd(f"docker kill {inst_name}")
            except Exception:
                pass
            finally:
                raise

    def _image_exists(self, image):
        """Implements a method to see if the container image is available"""
        output, _ = run_cmd(f"docker images -q {image}")
        log.debug("docker image id: %s", output)
        if re.match(r'^[0-9a-f]+$', output.strip()):
            return True
        try:
            run_cmd(f"docker pull {image}", _live_output_=True)
            return True
        except Exception:
            pass
        return False

    def _run_container_build_command(self, image, cache):
        log.info(f"Building docker image {image}.")
        run_cmd("docker build {options} {build_extra_options} -t {container} -f {script} .".format(
            options="" if cache else "--no-cache",
            container=image,
            script=self._script,
            build_extra_options=rconfig().docker.build_extra_options,
        ),
            _live_output_=rconfig().setup.live_output,
            _activity_timeout_=rconfig().setup.activity_timeout,
        )
        log.info(f"Successfully built docker image {image}.")

    def _upload_image(self, image):
        log.info(f"Publishing docker image {image}.")
        run_cmd(f"docker login && docker push {image}")
        log.info(f"Successfully published docker image {image}.")

    def _generate_script(self, custom_commands):
        docker_content = """FROM ubuntu:22.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update
RUN apt-get -y install apt-utils dialog locales tzdata sudo
RUN apt-get -y install curl wget unzip git
RUN apt-get -y install software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get -y install python{pyv} python{pyv}-venv python{pyv}-dev python3-pip
RUN apt-get -y install libhdf5-serial-dev
#RUN update-alternatives --install /usr/bin/python3 python3 $(which python{pyv}) 1

RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# aliases for the python system
ENV SPIP python{pyv} -m pip
ENV SPY python{pyv}

# Enforce UTF-8 encoding
ENV PYTHONUTF8 1
ENV PYTHONIOENCODING utf-8
# RUN locale-gen en-US.UTF-8
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Setup HDF5 for installing `tables`
ENV HDF5_DIR /usr/lib/aarch64-linux-gnu/hdf5/serial

WORKDIR /bench

# We create a virtual environment so that AutoML systems may use their preferred versions of
# packages that we need to data pre- and postprocessing without breaking it.
RUN $SPIP install -U pip wheel
RUN $SPY -m venv venv
ENV PIP /bench/venv/bin/python{pyv} -m pip
ENV PY /bench/venv/bin/python{pyv} -W ignore
#RUN $PIP install -U pip=={pipv} wheel
RUN $PIP install -U pip wheel

VOLUME /input
VOLUME /output
VOLUME /custom

# Add the AutoML system except files listed in .dockerignore (could also use git clone directly?)
ADD . /bench/

RUN (grep -v '^\\s*#' | xargs -L 1 $PIP install --no-cache-dir) < requirements.txt

RUN $PY {script} {framework} -s only
{custom_commands}

# https://docs.docker.com/engine/reference/builder/#entrypoint
ENTRYPOINT ["/bin/bash", "-c", "$PY {script} $0 $*"]
CMD ["{framework}", "test"]

""".format(
            custom_commands=custom_commands.format(
                setup=dir_of(os.path.join(self._framework_dir, "setup", ""),
                             rel_to_project_root=True),
                pip="$PIP",
                py="$PY"
            ),
            framework=self._forward_params['framework_name'],
            pyv=rconfig().versions.python,
            pipv=rconfig().versions.pip,
            script=rconfig().script,
        )

        touch(self._script)
        with open(self._script, 'w') as file:
            file.write(docker_content)


def resolve_docker_run_as_option(option: str) -> str:
    """ Resolve `docker.run_as` option into the correct `-u` option for `docker run`.

    option, str: one of 'user' (unix only), 'root', 'default', or a valid `-u` option.
               * 'user': set as `-u $(id -u):$(id -g)`, only on unix systems.
               * 'root': set as `-u 0:0`
               * 'default': does not set `-u`
               * any string that starts with `-u`, which will be directly forwarded.

    For linux specifically, files created within docker are *not always*
    automatically owned by the user starting the docker instance.
    We had reports of different behavior even for people running the same OS and Docker.
    """
    if option == "default":
        return ''
    if option == "root":
        return '-u 0:0'
    if option == "user":
        if os.name == 'nt':
            raise ValueError("docker.run_as: 'user' is not supported on Windows.")
        return f'-u "{os.getuid()}:{os.getgid()}"'
    if option.startswith("-u"):
        return rconfig().docker.run_as
    raise ValueError(f"Invalid setting for `docker.run_as`: '{option}'.")
