import logging
import os
import re

from .benchmark import Benchmark
from .resources import config as rconfig
from .results import Scoreboard
from .utils import run_cmd


log = logging.getLogger(__name__)


class DockerBenchmark(Benchmark):
    """DockerBenchmark
    an extension of Benchmark to run benchmarks inside docker.
    """

    @staticmethod
    def docker_image_name(framework_def):
        docker_image = framework_def.docker_image
        return "{author}/{image}:{tag}".format(
            author=docker_image["author"],
            image=docker_image["image"],
            tag=docker_image["tag"]
        )

    def __init__(self, framework_name, benchmark_name, keep_instance=False):
        """

        :param framework_name:
        :param benchmark_name:
        :param keep_instance:
        """
        super().__init__(framework_name, benchmark_name)
        self.keep_instance = keep_instance

    def setup(self, mode, upload=False):
        if mode == Benchmark.SetupMode.skip:
            return

        if mode == Benchmark.SetupMode.auto and self._docker_image_exists():
            return

        custom_commands = self.framework_module.docker_commands() if hasattr(self.framework_module, 'docker_commands') else ""
        self._generate_docker_script(custom_commands)
        self._build_docker_image()
        if upload:
            self._upload_docker_image()

    def cleanup(self):
        # todo: remove generated docker script? anything else?
        pass

    def run(self, save_scores=False):
        if self.keep_instance:
            self._start_docker("{framework} {benchmark}".format(
                framework=self.framework_def.name,
                benchmark=self.benchmark_name
            ))
            return
        else:
            return super().run(save_scores=False)

    def run_one(self, task_name: str, fold, save_scores=False):
        if self.keep_instance and (fold is None or (isinstance(fold, list) and len(fold) > 1)):
            self._start_docker("{framework} {benchmark} -t {task} {folds}".format(
                framework=self.framework_def.name,
                benchmark=self.benchmark_name,
                task=task_name,
                folds='' if fold is None else ' '.join(['-f']+fold)
            ))
        else:
            return super().run_one(task_name=task_name, fold=fold, save_scores=False)
        # board = Scoreboard.for_task(task_name, framework_name=self.framework_name)

    def _run_task(self, task_def):
        if self.keep_instance:
            self._start_docker("{framework} {benchmark} -t {task}".format(
                framework=self.framework_def.name,
                benchmark=self.benchmark_name,
                task=task_def.name,
            ))
        else:
            super()._run_task(task_def)
        return []

    def _run_fold(self, task_def, fold: int):
        self._start_docker("{framework} {benchmark} -t {task} -f {fold}".format(
            framework=self.framework_def.name,
            benchmark=self.benchmark_name,
            task=task_def.name,
            fold=fold
        ))

    def _start_docker(self, script_params=""):
        in_dir = rconfig().input_dir
        out_dir = rconfig().output_dir
        cmd = "docker run -v {input}:/input -v {output}:/output --rm {image} {params} -i /input -o /output -s skip".format(
            input=in_dir,
            output=out_dir,
            image=self._docker_image_name,
            params=script_params
        )
        log.info("Starting docker: %s", cmd)
        log.info("Datasets are loaded by default from folder %s", in_dir)
        log.info("Generated files will be available in folder %s", out_dir)
        output = run_cmd(cmd)
        log.debug(output)

    @property
    def _docker_script(self):
        return os.path.join(self._framework_dir, 'Dockerfile')

    @property
    def _docker_image_name(self):
        return DockerBenchmark.docker_image_name(self.framework_def)

    def _docker_image_exists(self):
        output = run_cmd("docker images -q {image}".format(image=self._docker_image_name))
        log.debug("docker image id: %s", output)
        return re.match(r'^[0-9a-f]+$', output.strip())

    def _build_docker_image(self):
        log.info("Building docker image %s", self._docker_image_name)
        output = run_cmd("docker build -t {container} -f {script} .".format(
            container=self._docker_image_name,
            script=self._docker_script
        ))
        log.info("Successfully built docker image %s", self._docker_image_name)
        log.debug(output)

    def _upload_docker_image(self):
        log.info("Publishing docker image %s", self._docker_image_name)
        output = run_cmd("docker login && docker push {}".format(self._docker_image_name))
        log.info("Successfully published docker image %s", self._docker_image_name)
        log.debug(output)

    def _generate_docker_script(self, custom_commands):
        docker_content = """FROM ubuntu:18.04

RUN apt-get update
RUN apt-get install -y curl wget unzip git
RUN apt-get install -y python3 python3-pip python3-venv
RUN pip3 install --upgrade pip

# We create a virtual environment so that AutoML systems may use their preferred versions of 
# packages that we need to data pre- and postprocessing without breaking it.
ENV V_PIP /venvs/setup/bin/pip3
ENV V_PY /venvs/setup/bin/python3
ENV PIP pip3
ENV PY python3 -W ignore

RUN $PY -m venv /venvs/setup
RUN $V_PIP install --upgrade pip

WORKDIR /bench
VOLUME /input
VOLUME /output

# Add the AutoML system except files listed in .dockerignore (could also use git clone directly?)
ADD . /bench/

RUN $PIP install --no-cache-dir -r requirements.txt --process-dependency-links
RUN $PIP install --no-cache-dir openml

{custom_commands}

# https://docs.docker.com/engine/reference/builder/#entrypoint
ENTRYPOINT ["/bin/bash", "-c", "$PY {script} $0 $*"]
CMD ["{framework}", "test"]

""".format(custom_commands=custom_commands,
           framework=self.framework_name,
           script=rconfig().script)

        with open(self._docker_script, 'w') as file:
            file.write(docker_content)

