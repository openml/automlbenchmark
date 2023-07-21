"""
**container** module is build on top of **benchmark** module to provide logic to create and run container images (e.g. docker, singularity)
that are preconfigured with a given automl framework, and that can be used to run a benchmark anywhere.
The image embeds a version of the automlbenchmark app so that tasks are later run in local mode inside the container,
providing the same parameters and features allowing to import config and export results through mounted folders.
"""
from __future__ import annotations

from abc import abstractmethod
import logging
import re
from typing import List, Union

from ..benchmark import Benchmark, SetupMode
from ..errors import InvalidStateError
from ..job import Job
from ..resources import config as rconfig, get as rget
from ..__version__ import __version__, _dev_version as dev


log = logging.getLogger(__name__)


class ContainerBenchmark(Benchmark):
    """ContainerBenchmark
    an extension of Benchmark to run benchmarks inside a container.
    """
    framework_install_required = False

    @classmethod
    def image_name(cls, framework_def, label=None, **kwargs):
        if label is None:
            label = rget().project_info.branch

        di = framework_def.image
        author = di.author
        image = di.image if di.image else framework_def.name.lower()
        tags = [di.tag if di.tag else framework_def.version.lower()]
        if label not in rconfig().container.ignore_labels:
            tags.append(label)
        tag = re.sub(r"([^\w.-])", '.', '-'.join(tags))
        # Some frameworks allow specifying a version by #HASH which would lead to
        # the tag starting with a '.' which is invalid.
        tag = tag.lstrip('.')
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
        self.image = None

    def _container_image_name(self, label=None):
        return self.image_name(self.framework_def, label)

    def _validate(self):
        max_parallel_jobs = rconfig().job_scheduler.max_parallel_jobs
        if self.parallel_jobs == 0 or self.parallel_jobs > max_parallel_jobs:
            log.warning("Forcing parallelization to its upper limit: %s.", max_parallel_jobs)
            self.parallel_jobs = max_parallel_jobs

    def setup(self, mode, upload=False):
        if mode == SetupMode.skip:
            return

        if mode == SetupMode.auto:
            self.image = self._find_image()
            if self.image:
                return

        self._generate_script(self.custom_commands)
        self.image = self._build_image(cache=(mode != SetupMode.force))
        if upload:
            self._upload_image(self.image)

    def cleanup(self):
        # TODO: remove generated script? anything else?
        pass

    def run(self, tasks: str | list[str] | None = None, folds: int | list[int] | None = None):
        self._get_task_defs(tasks)  # validates tasks
        if self.parallel_jobs > 1 or not self.minimize_instances:
            return super().run(tasks, folds)
        else:
            job = self._make_container_job(tasks, folds)
            try:
                results = self._run_jobs([job])
                scoreboard = self._process_results(results)
                return self._results_summary(scoreboard)
            finally:
                self.cleanup()

    def _make_job(self, task_def, fold=int):
        return self._make_container_job([task_def.name], [fold]) if not self._skip_job(task_def, fold) else None

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
                ','.join(task_names) if len(task_names) > 0 else 'all_tasks',
                ','.join(folds) if len(folds) > 0 else 'all_folds',
                self.framework_name
            ]),
            raise_on_failure=rconfig().job_scheduler.exit_on_job_failure
        )
        job._run = _run
        return job

    def _start_container(self, script_params=""):
        """Implementes the container run method"""
        raise NotImplementedError

    def _find_image(self):
        images_lookup = ([self._custom_image_name] if self._custom_image_name
                         else [self._container_image_name(dev), self._container_image_name()] if __version__ == dev
                         else [self._container_image_name()])

        for image in images_lookup:
            if self._image_exists(image):
                return image
        return None

    def _image_exists(self, image):
        """Implements a method to see if the container image is available"""
        raise NotImplementedError

    def _build_image(self, cache=True):
        image = self._custom_image_name
        if self.force_branch:
            current_branch = rget().git_info.branch
            create_dev_image = False
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
                create_dev_image = True

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
                create_dev_image = True
            if create_dev_image and not image:
                image = self._container_image_name(dev)

        if not image:
            tags = rget().git_info.tags
            version_tags = [t for t in tags if re.match(r"v\d+(\d+.)*", t)]
            if len(version_tags) > 1:
                raise InvalidStateError(
                    "The image can't be built as more than one version tag was found."
                    f"Found tags: {version_tags}"
                )
            version_tag = next(iter(version_tags), None)
            image = self._container_image_name(version_tag)
        self._run_container_build_command(image, cache)
        return image

    def _run_container_build_command(self, image, cache):
        """Implements a method to build a container image"""
        raise NotImplementedError

    def _upload_image(self, image):
        """Implements a method to upload images to hub"""
        raise NotImplementedError

    def _generate_script(self, custom_commands):
        """Implements a method to create the recipe for a container script"""
        raise NotImplementedError
