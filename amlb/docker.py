"""
**docker** module is build on top of **ContainerBenchmark** module to provide logic to create and run docker images
that are preconfigured with a given automl framework, and that can be used to run a benchmark anywhere.
The docker image embeds a version of the automlbenchmark app so that tasks are later run in local mode inside docker,
providing the same parameters and features allowing to import config and export results through mounted folders.
"""
import logging
import os
import re

from amlb.container import ContainerBenchmark
from amlb.resources import config as rconfig
from amlb.utils import dir_of, run_cmd, touch


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
        return os.path.join(self._framework_dir, 'Dockerfile')

    def _start_container(self, script_params=""):
        """Implementes the container run method"""
        in_dir = rconfig().input_dir
        out_dir = rconfig().output_dir
        custom_dir = rconfig().user_dir
        for d in [in_dir, out_dir, custom_dir]:
            touch(d, as_dir=True)
        script_extra_params = ""
        inst_name = self.sid
        cmd = (
            "docker run --name {name} {options} "
            "-v {input}:/input -v {output}:/output -v {custom}:/custom "
            "--rm {image} {params} -i /input -o /output -u /custom -s skip -Xrun_mode=docker {extra_params}"
        ).format(
            name=inst_name,
            options=rconfig().docker.run_extra_options,
            input=in_dir,
            output=out_dir,
            custom=custom_dir,
            image=self._image_name,
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

    def _image_exists(self):
        """Implements a method to see if the container image is available"""
        output, _ = run_cmd(f"docker images -q {self._image_name}")
        log.debug("docker image id: %s", output)
        if re.match(r'^[0-9a-f]+$', output.strip()):
            return True
        try:
            run_cmd(f"docker pull {self._image_name}", _live_output_=True)
            return True
        except Exception:
            pass
        return False

    def _run_container_build_command(self, cache):
        image = self._image_name
        log.info(f"Building docker image {image}.")
        run_cmd("docker build {options} -t {container} -f {script} .".format(
            options="" if cache else "--no-cache",
            container=image,
            script=self._script),
            _live_output_=rconfig().setup.live_output,
            _activity_timeout_=rconfig().setup.activity_timeout
        )
        log.info(f"Successfully built docker image {image}.")

    def _upload_image(self):
        image = self._image_name
        log.info(f"Publishing docker image {image}.")
        run_cmd(f"docker login && docker push {image}")
        log.info(f"Successfully published docker image {image}.")

    def _generate_script(self, custom_commands):
        docker_content = """FROM ubuntu:18.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update
RUN apt-get -y install apt-utils dialog locales
RUN apt-get -y install curl wget unzip git
RUN apt-get -y install python3 python3-pip python3-venv
RUN pip3 install -U pip wheel

# aliases for the python system
ENV SPIP pip3
ENV SPY python3

# Enforce UTF-8 encoding
ENV PYTHONUTF8 1
ENV PYTHONIOENCODING utf-8
# RUN locale-gen en-US.UTF-8
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

WORKDIR /bench

# We create a virtual environment so that AutoML systems may use their preferred versions of
# packages that we need to data pre- and postprocessing without breaking it.
RUN $SPY -m venv venv
ENV PIP /bench/venv/bin/pip3
ENV PY /bench/venv/bin/python3 -W ignore
#RUN $PIP install -U pip=={pip_version} wheel
RUN $PIP install -U pip wheel

VOLUME /input
VOLUME /output
VOLUME /custom

# Add the AutoML system except files listed in .dockerignore (could also use git clone directly?)
ADD . /bench/

RUN xargs -L 1 $PIP install --no-cache-dir < requirements.txt

RUN $PY {script} {framework} -s only
{custom_commands}

# https://docs.docker.com/engine/reference/builder/#entrypoint
ENTRYPOINT ["/bin/bash", "-c", "$PY {script} $0 $*"]
CMD ["{framework}", "test"]

""".format(
            custom_commands=custom_commands.format(**dict(setup=dir_of(os.path.join(self._framework_dir, "setup/"),
                                                                       rel_to_project_root=True),
                                                          pip="$PIP",
                                                          py="$PY")),
            framework=self.framework_name,
            pip_version=rconfig().versions.pip,
            script=rconfig().script,
            user=rconfig().user_dir,
        )
        with open(self._script, 'w') as file:
            file.write(docker_content)
