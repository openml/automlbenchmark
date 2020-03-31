"""
**singularity** module is build on top of **benchmark** module to provide logic to create and run docker images
that are preconfigured with a given automl framework, and that can be used to run a benchmark anywhere.
The docker image embeds a version of the automlbenchmark app so that tasks are later run in local mode inside docker,
providing the same parameters and features allowing to import config and export results through mounted folders.
"""
import logging
import os
import re

from .benchmark import Benchmark, SetupMode
from .errors import InvalidStateError
from .job import Job
from .resources import config as rconfig, get as rget
from .utils import dir_of, run_cmd


log = logging.getLogger(__name__)


class SingularityBenchmark(Benchmark):
    """SingularityBenchmark
    an extension of Benchmark to run benchmarks inside docker.
    """

    @staticmethod
    def singularity_image_name(framework_def, branch=None):
        di = framework_def.docker_image
        if branch is None:
            branch = rget().project_info.branch
        return "{author}-{image}-{tag}".format(
            author=di.author,
            image=di.image if di.image else framework_def.name.lower(),
            tag=re.sub(r"([^\w.-])", '.',
                       '-'.join([di.tag if di.tag else framework_def.version.lower(), branch]))
        )

    def __init__(self, framework_name, benchmark_name, constraint_name):
        """

        :param framework_name:
        :param benchmark_name:
        :param constraint_name:
        """
        super().__init__(framework_name, benchmark_name, constraint_name)
        self._custom_image_name = rconfig().singularity.image

    def _validate(self):
        if self.parallel_jobs == 0 or self.parallel_jobs > rconfig().max_parallel_jobs:
            log.warning("Forcing parallelization to its upper limit: %s.", rconfig().max_parallel_jobs)
            self.parallel_jobs = rconfig().max_parallel_jobs

    def setup(self, mode, upload=False):
        if mode == SetupMode.skip:
            return

        if mode == SetupMode.auto and self._singularity_image_exists():
            log.info("Pre-existing SIF file found. Skip build...")
            return

        custom_commands = self.framework_module.singularity_commands(
            self.framework_def.setup_args,
            setup_cmd=self.framework_def._setup_cmd
        ) if hasattr(self.framework_module, 'singularity_commands') else ""

        self._generate_singularity_script(custom_commands)

        self._build_singularity_image(cache=(mode != SetupMode.force))
        if upload:
            self._upload_singularity_image()

    def cleanup(self):
        # TODO: remove generated docker script? anything else?
        pass

    def run(self, task_name=None, fold=None):
        self._get_task_defs(task_name)  # validates tasks
        if self.parallel_jobs > 1 or not rconfig().singularity.minimize_instances:
            return super().run(task_name, fold)
        else:
            job = self._make_singularity_job(task_name, fold)
            try:
                results = self._run_jobs([job])
                return self._process_results(results, task_name=task_name)
            finally:
                self.cleanup()

    def _make_job(self, task_def, fold=int):
        return self._make_singularity_job([task_def.name], [fold])

    def _make_singularity_job(self, task_names=None, folds=None):
        task_names = [] if task_names is None else task_names
        folds = [] if folds is None else [str(f) for f in folds]

        def _run():
            self._start_singularity("{benchmark} {constraint} {task_param} {folds_param} -Xseed={seed}".format(
                framework=self.framework_name,
                benchmark=self.benchmark_name,
                constraint=self.constraint_name,
                task_param='' if len(task_names) == 0 else ' '.join(['-t']+task_names),
                folds_param='' if len(folds) == 0 else ' '.join(['-f']+folds),
                seed=rget().seed(int(folds[0])) if len(folds) == 1 else rconfig().seed,
            ))
            # TODO: would be nice to reload generated scores and return them

        job = Job('_'.join(['singularity',
                            self.benchmark_name,
                            self.constraint_name,
                            '.'.join(task_names) if len(task_names) > 0 else 'all',
                            '.'.join(folds),
                            self.framework_name]))
        job._run = _run
        return job

    def _start_singularity(self, script_params=""):
        in_dir = rconfig().input_dir
        out_dir = rconfig().output_dir
        custom_dir = rconfig().user_dir
        script_extra_params = ""
        inst_name = self.sid
        cmd = (
            "singularity run {options} "
            "-B {input}:/input -B {output}:/output -B {custom}:/custom "
            "{image} \"{params} -i /input -o /output -u /custom -s skip -Xrun_mode=singularity {extra_params}\""
        ).format(
            name=inst_name,
            options=rconfig().singularity.run_extra_options,
            input=in_dir,
            output=out_dir,
            custom=custom_dir,
            image=self._singularity_image,
            params=script_params,
            extra_params=script_extra_params,
        )
        log.info("Starting singularity: %s.", cmd)
        log.info("Datasets are loaded by default from folder %s.", in_dir)
        log.info("Generated files will be available in folder %s.", out_dir)
        try:
            run_cmd(cmd, _capture_error_=False)  # console logs are written on stderr by default: not capturing allows live display
        except:  # also want to handle KeyboardInterrupt
            try:
                raise NotImplementedError
            except:
                pass
            finally:
                raise

    @property
    def _singularity_script(self):
        return os.path.join(self._framework_dir, 'Singularityfile')

    @property
    def _singularity_image_name(self):
        return self._custom_image_name or SingularityBenchmark.singularity_image_name(self.framework_def)

    @property
    def _singularity_image(self):
        # By default, when the image is created the autoMLbenchmar dir is added
        # to the container. Created and read sif files from /tmp and not from
        # run/ dir
        return os.path.join('/tmp', self._singularity_image_name + ".sif")

    def _singularity_image_exists(self):
        # In comparisson to Docker, the singularity image is just a file
        if os.path.exists(self._singularity_image):
            log.debug("Singularity image found on: %s", self._singularity_image)
            return True
        else:

            try:
                run_cmd("singularity pull {image} library://{library}/{image_name}:latest".format(
                    image=self._singularity_image,
                    image_name=os.path.basename(self._singularity_image),
                    library=rconfig().singularity.library)
                )
                return True
            except:
                pass
        return False

    def _build_singularity_image(self, cache=True):
        if rconfig().singularity.force_branch:
            run_cmd("git fetch")
            current_branch = run_cmd("git rev-parse --abbrev-ref HEAD")[0].strip()
            status, _ = run_cmd("git status -b --porcelain")
            if len(status.splitlines()) > 1 or re.search(r'\[(ahead|behind) \d+\]', status):
                log.info("Branch status:\n%s", status)
                force = None
                while force not in ['y', 'n']:
                    force = input(f"""Branch `{current_branch}` is not clean or up-to-date.
Do you still want to build the singularity image? (y/[n]) """).lower() or 'n'
                if force == 'n':
                    raise InvalidStateError(
                        "Singularity image can't be built as the current branch is not clean or up-to-date. "
                        "Please switch to the expected `{}` branch, and ensure that it is clean before building the singularity image.".format(rget().project_info.branch)
                    )

            tag = rget().project_info.tag
            tags, _ = run_cmd("git tag --points-at HEAD")
            if tag and not re.search(r'(?m)^{}$'.format(tag), tags):
                force = None
                while force not in ['y', 'n']:
                    force = input(f"""Branch `{current_branch}` isn't tagged as `{tag}` (as required by config.project_repository).
Do you still want to build the singularity image? (y/[n]) """).lower() or 'n'
                if force == 'y':
                    self._custom_image_name = self._custom_image_name or SingularityBenchmark.singularity_image_name(self.framework_def, current_branch)
                else:
                    raise InvalidStateError(
                        "Singularity image can't be built as current branch is not tagged as required `{}`. "
                        "Please switch to the expected tagged branch before building the singularity image.".format(tag)
                    )

        log.info(f"Building singularity image {self._singularity_image}.")
        # Remote last too much so that jobs get killed 5e7a4bc490b784c737d095a5 build exceeded max build timeFATAL:   While performing build: build has not completed
        run_cmd("sudo singularity build {options} {container} {script}".format(
            options="" if cache else "--disable-cache",
            container=self._singularity_image,
            script=self._singularity_script
        ), _live_output_=True)
        # Create also the container for push
        log.info(f"Successfully built singularity image {self._singularity_image}.")


    def _upload_singularity_image(self):
        image = self._singularity_image_name
        library=rconfig().singularity.library
        log.info(f"Publishing Singularity image {image}.")
        run_cmd(f"singularity login && singularity push -U {self._singularity_image} library://{library}/{os.path.basename(self._singularity_image)}:latest")
        log.info(f"Successfully published singularity image {image}.")


        raise NotImplementedError

    def _generate_singularity_script(self, custom_commands):
        singularity_content="""Bootstrap: docker
From: ubuntu:18.04
%files
. /bench/
%post

DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get -y install apt-utils dialog locales
apt-get -y install curl wget unzip git
apt-get -y install python3 python3-pip python3-venv
pip3 install -U pip

# aliases for the python system
SPIP=pip3
SPY=python3

# Enforce UTF-8 encoding
PYTHONUTF8=1
PYTHONIOENCODING=utf-8
# RUN locale-gen en-US.UTF-8
LANG=C.UTF-8
LC_ALL=C.UTF-8

cd /bench

# We create a virtual environment so that AutoML systems may use their preferred versions of
# packages that we need to data pre- and postprocessing without breaking it.
$SPY -m venv venv
PIP=/bench/venv/bin/pip3
PY=/bench/venv/bin/python3
#RUN $PIP install -U pip=={pip_version}
$PIP install -U pip

mkdir /input
mkdir /output
mkdir /custom

# Add the AutoML system except files listed in .dockerignore (could also use git clone directly?)

xargs -L 1 $PIP install --no-cache-dir < requirements.txt

{custom_commands}

%environment
export DEBIAN_FRONTEND=noninteractive
export SPIP=pip3
export SPY=python3
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PIP=/bench/venv/bin/pip3
export PY=/bench/venv/bin/python3
%runscript
cd /bench
exec /bin/bash -c "$PY {script} {framework} ""$@"
%startscript
cd /bench
exec /bin/bash -c "$PY {script} {framework} ""$@"

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

        with open(self._singularity_script, 'w') as file:
            file.write(singularity_content)

