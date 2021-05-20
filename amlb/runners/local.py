"""
**main** module handles all the main running logic:

- load specified framework and benchmark.
- extract the tasks and configure them.
- create jobs for each task.
- run the jobs.
- collect and save results.
"""
from copy import copy
from importlib import import_module, invalidate_caches
import logging
import os
import sys
from typing import List, Union

from ..benchmark import SetupMode, Benchmark, TaskConfig
from ..data import DatasetType
from ..datasets import DataSourceType
from ..job import Job
from ..resources import get as rget, config as rconfig
from ..results import ErrorResult, TaskResult
from ..utils import Namespace as ns, as_list, json_dump, profile, repr_def, run_cmd, run_script, touch


log = logging.getLogger(__name__)


__installed_file__ = '.installed'
__setup_env_file__ = '.setup_env'


class LocalBenchmark(Benchmark):
    """Benchmark.
    Structure containing the generic information needed to run a benchmark:
     - the datasets
     - the automl framework


     we need to support:
     - openml tasks
     - openml datasets
     - openml studies (=benchmark suites)
     - user-defined (list of) datasets
    """

    data_loader = None
    run_mode = 'local'

    def __init__(self, framework_name: str, benchmark_name: str, constraint_name: str):
        super().__init__(framework_name, benchmark_name, constraint_name)
        self.framework_module = import_module(self.framework_def.module)

    def _validate(self):
        if self.parallel_jobs > 1:
            log.warning("Parallelization is not supported in local mode: ignoring `parallel_jobs=%s` parameter.", self.parallel_jobs)
            self.parallel_jobs = 1

    def setup(self, mode: SetupMode):
        """
        ensure all dependencies needed by framework are available
        and possibly download them if necessary.
        Delegates specific setup to the framework module
        """
        super().setup(mode)
        if mode == SetupMode.skip or mode == SetupMode.auto and self._is_setup_done():
            return

        log.info("Setting up framework {}.".format(self.framework_name))

        self._write_setup_env(self.framework_module.__path__[0], **dict(self.framework_def.setup_env))
        self._mark_setup_start()

        if hasattr(self.framework_module, 'setup'):
            self.framework_module.setup(*self.framework_def.setup_args,
                                        _live_output_=rconfig().setup.live_output,
                                        _activity_timeout_=rconfig().setup.activity_timeout)

        if self.framework_def.setup_script is not None:
            run_script(self.framework_def.setup_script,
                       _live_output_=rconfig().setup.live_output,
                       _activity_timeout_=rconfig().setup.activity_timeout)

        if self.framework_def.setup_cmd is not None:
            def resolve_venv(cmd):
                venvs = [
                    *[os.path.join(p, "venv") for p in self.framework_module.__path__],
                    os.path.join(rconfig().root_dir, "venv"),
                ]
                venv = next((ve for ve in venvs if os.path.isdir(ve)), None)
                py = os.path.join(venv, "bin", "python") if venv else "python"
                pip = os.path.join(venv, "bin", "pip") if venv else "pip"
                return cmd.format(py=py, pip=pip)

            setup_cmd = [resolve_venv(cmd) for cmd in self.framework_def.setup_cmd]
            run_cmd('\n'.join(setup_cmd),
                    _executable_="/bin/bash",
                    _live_output_=rconfig().setup.live_output,
                    _activity_timeout_=rconfig().setup.activity_timeout)

        invalidate_caches()
        log.info("Setup of framework {} completed successfully.".format(self.framework_name))

        self._mark_setup_done()

    def _write_setup_env(self, dest_dir, **kwargs):
        setup_env = dict(
            AMLB_ROOT=rconfig().root_dir,
            PY_EXEC_PATH=sys.executable
        )
        setup_env.update(**kwargs)
        with open(os.path.join(dest_dir, __setup_env_file__), 'w') as f:
            f.write('\n'.join([f"{k}={v}" for k, v in setup_env.items()]+[""]))

    def _installed_file(self):
        return os.path.join(self._framework_dir, __installed_file__)

    def _installed_version(self):
        installed = self._installed_file()
        versions = []
        if os.path.isfile(installed):
            with open(installed, 'r') as f:
                versions = list(filter(None, map(str.strip, f.readlines())))
        return versions

    def _is_setup_done(self):
        return self.framework_def.version in self._installed_version()

    def _mark_setup_start(self):
        installed = self._installed_file()
        if os.path.isfile(installed):
            os.remove(installed)

    def _mark_setup_done(self):
        installed = self._installed_file()
        versions = []
        if hasattr(self.framework_module, 'version'):
            versions.append(self.framework_module.version())
        versions.extend([self.framework_def.version, ""])
        with open(installed, 'a') as f:
            f.write('\n'.join(versions))

    def run(self, tasks: Union[str, List[str]] = None, folds: Union[int, List[int]] = None):
        assert self._is_setup_done(), f"Framework {self.framework_name} [{self.framework_def.version}] is not installed."
        super().run(tasks, folds)

    def _make_job(self, task_def, fold: int):
        """
        runs the framework against a given fold
        :param task_def: the task to run
        :param fold: the specific fold to use on this task
        """
        if fold < 0 or fold >= task_def.folds:
            # raise ValueError(f"Fold value {fold} is out of range for task {task_def.name}.")
            log.warning(f"Fold value {fold} is out of range for task {task_def.name}, skipping it.")
            return

        return LocalTask(self, task_def, fold).as_job()

    @property
    def _framework_dir(self):
        return os.path.dirname(self.framework_module.__file__)


class LocalTask:

    def __init__(self, benchmark: LocalBenchmark, task_def, fold):
        """

        :param task_def:
        :param fold:
        """
        self.benchmark = benchmark
        self._task_def = task_def
        self.fold = fold
        self.task_config = TaskConfig(
            name=task_def.name,
            fold=fold,
            metrics=task_def.metric,
            seed=rget().seed(fold),
            max_runtime_seconds=task_def.max_runtime_seconds,
            cores=task_def.cores,
            max_mem_size_mb=task_def.max_mem_size_mb,
            min_vol_size_mb=task_def.min_vol_size_mb,
            input_dir=rconfig().input_dir,
            output_dir=benchmark.output_dirs.session,
            run_mode=rconfig().run_mode
        )
        # allowing to override some task parameters through command line, e.g.: -Xt.max_runtime_seconds=60
        if rconfig()['t'] is not None:
            for c in dir(self.task_config):
                if rconfig().t[c] is not None:
                    setattr(self.task_config, c, rconfig().t[c])
        self._dataset = None

    def load_data(self):
        """
        Loads the training dataset for the current given task
        :return: path to the dataset file
        """
        if hasattr(self._task_def, 'openml_task_id'):
            self._dataset = self.benchmark.data_loader.load(DataSourceType.openml_task, task_id=self._task_def.openml_task_id, fold=self.fold)
            log.debug("Loaded OpenML dataset for task_id %s.", self._task_def.openml_task_id)
        elif hasattr(self._task_def, 'openml_dataset_id'):
            # TODO
            raise NotImplementedError("OpenML datasets without task_id are not supported yet.")
        elif hasattr(self._task_def, 'dataset'):
            self._dataset = self.benchmark.data_loader.load(DataSourceType.file, dataset=self._task_def.dataset, fold=self.fold)
        else:
            raise ValueError("Tasks should have one property among [openml_task_id, openml_dataset_id, dataset].")

    def as_job(self):
        job = Job(name=rconfig().token_separator.join([
            'local',
            self.benchmark.benchmark_name,
            self.benchmark.constraint_name,
            self.task_config.name,
            str(self.fold),
            self.benchmark.framework_name
        ]),
            # specifying a job timeout to handle edge cases where framework never completes or hangs
            # (adding 5min safety to let the potential subprocess handle the interruption first).
            timeout_secs=self.task_config.job_timeout_seconds+5*60,
            raise_on_failure=rconfig().job_scheduler.exit_on_job_failure,
        )
        job._setup = self.setup
        job._run = self.run
        return job

    def setup(self):
        self.task_config.estimate_system_params()
        self.load_data()

    @profile(logger=log)
    def run(self):
        results = TaskResult(task_def=self._task_def, fold=self.fold,
                             constraint=self.benchmark.constraint_name,
                             predictions_dir=self.benchmark.output_dirs.predictions)
        framework_def = self.benchmark.framework_def
        task_config = copy(self.task_config)
        task_config.type = 'regression' if self._dataset.type == DatasetType.regression else 'classification'
        task_config.type_ = self._dataset.type.name
        task_config.framework = self.benchmark.framework_name
        task_config.framework_params = framework_def.params
        task_config.framework_version = self.benchmark._installed_version()[0]

        # allowing to pass framework parameters through command line, e.g.: -Xf.verbose=True -Xf.n_estimators=3000
        if rconfig()['f'] is not None:
            task_config.framework_params = ns.dict(ns(framework_def.params) + rconfig().f)

        task_config.output_predictions_file = results._predictions_file
        task_config.output_metadata_file = results._metadata_file
        touch(os.path.dirname(task_config.output_predictions_file), as_dir=True)
        if task_config.metrics is None:
            task_config.metrics = as_list(rconfig().benchmarks.metrics[self._dataset.type.name])
            task_config.metric = task_config.metrics[0]

        result = meta_result = None
        try:
            log.info("Running task %s on framework %s with config:\n%s", task_config.name, self.benchmark.framework_name, repr_def(task_config))
            json_dump(task_config, task_config.output_metadata_file, style='pretty')
            meta_result = self.benchmark.framework_module.run(self._dataset, task_config)
        except Exception as e:
            if rconfig().job_scheduler.exit_on_job_failure:
                raise
            log.exception(e)
            result = ErrorResult(e)
        finally:
            self._dataset.release()
        return results.compute_score(result=result, meta_result=meta_result)

