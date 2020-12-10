"""
**container** module is build on top of **benchmark** module to provide logic to create and run container images (e.g. docker, singularity)
that are preconfigured with a given automl framework, and that can be used to run a benchmark anywhere.
The image embeds a version of the automlbenchmark app so that tasks are later run in local mode inside the container,
providing the same parameters and features allowing to import config and export results through mounted folders.
"""
from abc import abstractmethod
import logging
import os
import re

from ..benchmark import Benchmark, SetupMode
from ..errors import InvalidStateError
from ..job import Job
from ..resources import config as rconfig, get as rget
from ..utils import dir_of, run_cmd
from ..__version__ import __version__ as dev


log = logging.getLogger(__name__)


class ContainerBenchmark(Benchmark):
    """ContainerBenchmark
    an extension of Benchmark to run benchmarks inside a container.
    """

    @classmethod
    def image_name(cls, framework_def, branch=None, **kwargs):
        if branch is None:
            branch = rget().project_info.branch

        di = framework_def.image
        author = di.author
        image = di.image if di.image else framework_def.name.lower()
        tags = [di.tag if di.tag else framework_def.version.lower()]
        if branch != 'master':
            tags.append(branch)
        tag = re.sub(r"([^\w.-])", '.', '-'.join(tags))
        return f"{author}/{image}:{tag}"

    @abstractmethod
    def __init__(self, framework_name, benchmark_name, constraint_name):
        """

        :param framework_name:
        :param benchmark_name:
        :param constraint_name:
        """
        super().__init__(framework_name, benchmark_name, constraint_name)
        self._custom_image_name = rconfig().container.image
        self.minimize_instances = rconfig().container.minimize_instances
        self.container_name = None
        self.force_branch = rconfig().container.force_branch
        self.custom_commands = ""

    def _container_image_name(self, branch=None):
        return self.image_name(self.framework_def, branch)

    def _validate(self):
        if self.parallel_jobs == 0 or self.parallel_jobs > rconfig().max_parallel_jobs:
            log.warning("Forcing parallelization to its upper limit: %s.", rconfig().max_parallel_jobs)
            self.parallel_jobs = rconfig().max_parallel_jobs

    def setup(self, mode, upload=False):
        if mode == SetupMode.skip:
            return

        if mode == SetupMode.auto and self._image_exists():
            return

        self._generate_script(self.custom_commands)
        self._build_image(cache=(mode != SetupMode.force))
        if upload:
            self._upload_image()

    def cleanup(self):
        # TODO: remove generated script? anything else?
        pass

    def run(self, task_name=None, fold=None):
        self._get_task_defs(task_name)  # validates tasks
        if self.parallel_jobs > 1 or not self.minimize_instances:
            return super().run(task_name, fold)
        else:
            job = self._make_container_job(task_name, fold)
            try:
                results = self._run_jobs([job])
                return self._process_results(results, task_name=task_name)
            finally:
                self.cleanup()

    def _make_job(self, task_def, fold=int):
        return self._make_container_job([task_def.name], [fold])

    def _make_container_job(self, task_names=None, folds=None):
        task_names = [] if task_names is None else task_names
        folds = [] if folds is None else [str(f) for f in folds]

        def _run():
            self._start_container("{framework} {benchmark} {constraint} {task_param} {folds_param} -Xseed={seed}".format(
                framework=self._forward_params['framework_name'],
                benchmark=self._forward_params['benchmark_name'],
                constraint=self._forward_params['constraint_name'],
                task_param='' if len(task_names) == 0 else ' '.join(['-t']+task_names),
                folds_param='' if len(folds) == 0 else ' '.join(['-f']+folds),
                seed=rget().seed(int(folds[0])) if len(folds) == 1 else rconfig().seed,
            ))
            # TODO: would be nice to reload generated scores and return them

        job = Job(rconfig().token_separator.join([
            self.container_name,
            self.benchmark_name,
            self.constraint_name,
            ' '.join(task_names) if len(task_names) > 0 else 'all',
            ' '.join(folds),
            self.framework_name
        ]))
        job._run = _run
        return job

    def _start_container(self, script_params=""):
        """Implementes the container run method"""
        raise NotImplementedError

    @property
    def _image_name(self):
        return self._custom_image_name or self._container_image_name()

    def _image_exists(self):
        """Implements a method to see if the container image is available"""
        raise NotImplementedError

    def _build_image(self, cache=True):
        if self.force_branch:
            current_branch = rget().git_info.branch
            create_custom_name = False
            status = rget().git_info.status
            if len(status) > 1 or re.search(r'\[(ahead|behind) \d+\]', status[0]):
                print("Branch status:\n%s", '\n'.join(status))
                force = None
                while force not in ['y', 'n']:
                    force = input(f"""Branch `{current_branch}` is not clean or up-to-date.
Do you still want to build the container image? (y/[n]) """).lower() or 'n'
                if force == 'n':
                    raise InvalidStateError(
                        "The image can't be built as the current branch is not clean or up-to-date. "
                        "Please switch to the expected `{}` branch, and ensure that it is clean before building the container image.".format(rget().project_info.branch)
                    )
                create_custom_name = True

            expected_branch = rget().project_info.branch
            tags = rget().git_info.tags
            if expected_branch and expected_branch not in tags+[current_branch]:
                force = None
                while force not in ['y', 'n']:
                    force = input(f"""Branch `{current_branch}` doesn't match `{expected_branch}` (as required by config.project_repository).
Do you still want to build the container image? (y/[n]) """).lower() or 'n'
                if force == 'n':
                    raise InvalidStateError(
                        "The image can't be built as current branch is not tagged as required `{}`. "
                        "Please switch to the expected tagged branch before building the container image.".format(expected_branch)
                    )
                create_custom_name = True
            if create_custom_name and not self._custom_image_name:
                self._custom_image_name = self._container_image_name(dev)

        self._run_container_build_command(cache)

    def _run_container_build_command(self, cache):
        """Implements a method to build a container image"""
        raise NotImplementedError

    def _upload_image(self):
        """Implements a method to upload images to hub"""
        raise NotImplementedError

    def _generate_script(self, custom_commands):
        """Implements a method to create the recipe for a container script"""
        raise NotImplementedError
