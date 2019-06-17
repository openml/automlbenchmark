"""
**docker** module is build on top of **benchmark** module to provide logic to create and run docker images
that are preconfigured with a given automl framework, and that can be used to run a benchmark anywhere.
The docker image embeds a version of the automlbenchmark app so that tasks are later run in local mode inside docker,
providing the same parameters and features allowing to import config and export results through mounted folders.
"""
import logging
import os
import re

from .benchmark import Benchmark
from .errors import InvalidStateError
from .job import Job
from .resources import config as rconfig, get as rget
from .utils import dir_of, run_cmd


log = logging.getLogger(__name__)


class DockerBenchmark(Benchmark):
    """DockerBenchmark
    an extension of Benchmark to run benchmarks inside docker.
    """

    @staticmethod
    def docker_image_name(framework_def):
        di = framework_def.docker_image
        return "{author}/{image}:{tag}".format(
            author=di.author,
            image=di.image if di.image else framework_def.name.lower(),
            tag='-'.join([di.tag if di.tag else framework_def.version.lower(), rget().project_info.branch])
        )

    def __init__(self, framework_name, benchmark_name, parallel_jobs=1):
        """

        :param framework_name:
        :param benchmark_name:
        :param parallel_jobs:
        """
        super().__init__(framework_name, benchmark_name, parallel_jobs)

    def _validate(self):
        if self.parallel_jobs == 0 or self.parallel_jobs > rconfig().max_parallel_jobs:
            log.warning("Forcing parallelization to its upper limit: %s.", rconfig().max_parallel_jobs)
            self.parallel_jobs = rconfig().max_parallel_jobs

    def setup(self, mode, upload=False):
        if mode == Benchmark.SetupMode.skip:
            return

        if mode == Benchmark.SetupMode.auto and self._docker_image_exists():
            return

        custom_commands = self.framework_module.docker_commands(
            self.framework_def.setup_args,
            setup_cmd=self.framework_def._setup_cmd
        ) if hasattr(self.framework_module, 'docker_commands') else ""
        self._generate_docker_script(custom_commands)
        self._build_docker_image(cache=(mode != Benchmark.SetupMode.force))
        if upload:
            self._upload_docker_image()

    def cleanup(self):
        # TODO: remove generated docker script? anything else?
        pass

    def run(self, task_name=None, fold=None):
        self._get_task_defs(task_name)  # validates tasks
        if self.parallel_jobs > 1 or not rconfig().docker.minimize_instances:
            return super().run(task_name, fold)
        else:
            job = self._make_docker_job(task_name, fold)
            try:
                results = self._run_jobs([job])
                return self._process_results(results, task_name=task_name)
            finally:
                self.cleanup()

    def _make_job(self, task_def, fold=int):
        return self._make_docker_job([task_def.name], [fold])

    def _make_docker_job(self, task_names=None, folds=None):
        task_names = [] if task_names is None else task_names
        folds = [] if folds is None else [str(f) for f in folds]

        def _run():
            self._start_docker("{framework} {benchmark} {task_param} {folds_param} -Xseed={seed}".format(
                framework=self.framework_name,
                benchmark=self.benchmark_name,
                task_param='' if len(task_names) == 0 else ' '.join(['-t']+task_names),
                folds_param='' if len(folds) == 0 else ' '.join(['-f']+folds),
                seed=rget().seed(int(folds[0])) if len(folds) == 1 else rconfig().seed,
            ))
            # TODO: would be nice to reload generated scores and return them

        job = Job('_'.join(['docker',
                            self.benchmark_name,
                            '.'.join(task_names) if len(task_names) > 0 else 'all',
                            '.'.join(folds),
                            self.framework_name]))
        job._run = _run
        return job

    def _start_docker(self, script_params=""):
        in_dir = rconfig().input_dir
        out_dir = rconfig().output_dir
        custom_dir = rconfig().user_dir
        script_extra_params = ""
        cmd = (
            "docker run {options} "
            "-v {input}:/input -v {output}:/output -v {custom}:/custom "
            "--rm {image} {params} -i /input -o /output -u /custom -s skip -Xrun_mode=docker {extra_params}"
        ).format(
            options=rconfig().docker.run_extra_options,
            input=in_dir,
            output=out_dir,
            custom=custom_dir,
            image=self._docker_image_name,
            params=script_params,
            extra_params=script_extra_params,
        )
        log.info("Starting docker: %s.", cmd)
        log.info("Datasets are loaded by default from folder %s.", in_dir)
        log.info("Generated files will be available in folder %s.", out_dir)
        run_cmd(cmd, _capture_error_=False)  # console logs are written on stderr by default: not capturing allows live display

    @property
    def _docker_script(self):
        return os.path.join(self._framework_dir, 'Dockerfile')

    @property
    def _docker_image_name(self):
        return DockerBenchmark.docker_image_name(self.framework_def)

    def _docker_image_exists(self):
        output, _ = run_cmd("docker images -q {image}".format(image=self._docker_image_name))
        log.debug("docker image id: %s", output)
        if re.match(r'^[0-9a-f]+$', output.strip()):
            return True
        try:
            run_cmd("docker pull {image}".format(image=self._docker_image_name))
            return True
        except:
            pass
        return False

    def _build_docker_image(self, cache=True):
        if rconfig().docker.force_branch:
            run_cmd("git fetch")
            status, _ = run_cmd("git status -b --porcelain")
            if len(status.splitlines()) > 1 or re.search(r'\[(ahead|behind) \d+\]', status):
                log.info("Branch status:\n%s", status)
                raise InvalidStateError("Docker image can't be built as the current branch is not up-to-date. "
                                        "Please switch to the expected up-to-date `{}` branch first.".format(rget().project_info.branch))

            tag = rget().project_info.tag
            tags, _ = run_cmd("git tag --points-at HEAD")
            if tag and not re.search(r'(?m)^{}$'.format(tag), tags):
                raise InvalidStateError("Docker image can't be built as current branch is not tagged as required `{}`. "
                                        "Please switch to the expected tagged branch first.".format(tag))

        log.info("Building docker image %s.", self._docker_image_name)
        run_cmd("docker build {options} -t {container} -f {script} .".format(
            options="" if cache else "--no-cache",
            container=self._docker_image_name,
            script=self._docker_script
        ))
        log.info("Successfully built docker image %s.", self._docker_image_name)

    def _upload_docker_image(self):
        log.info("Publishing docker image %s.", self._docker_image_name)
        run_cmd("docker login && docker push {}".format(self._docker_image_name))
        log.info("Successfully published docker image %s.", self._docker_image_name)

    def _generate_docker_script(self, custom_commands):
        docker_content = """FROM ubuntu:18.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update
RUN apt-get -y install apt-utils dialog locales
RUN apt-get -y install curl wget unzip git
RUN apt-get -y install python3 python3-pip python3-venv
RUN pip3 install -U pip

# We create a virtual environment so that AutoML systems may use their preferred versions of 
# packages that we need to data pre- and postprocessing without breaking it.
ENV PIP /venvs/bench/bin/pip3
ENV PY /venvs/bench/bin/python3 -W ignore
ENV SPIP pip3
ENV SPY python3

# Enforce UTF-8 encoding
ENV PYTHONUTF8 1
ENV PYTHONIOENCODING utf-8
# RUN locale-gen en-US.UTF-8
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN $SPY -m venv /venvs/bench
RUN $PIP install -U pip=={pip_version}

WORKDIR /bench
VOLUME /input
VOLUME /output
VOLUME /custom

# Add the AutoML system except files listed in .dockerignore (could also use git clone directly?)
ADD . /bench/

RUN xargs -L 1 $PIP install --no-cache-dir < requirements.txt

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
        with open(self._docker_script, 'w') as file:
            file.write(docker_content)

