"""
**benchmark** module handles all the main logic:

- load specified framework and benchmark.
- extract the tasks and configure them.
- create jobs for each task.
- run the jobs.
- collect and save results.
"""

from __future__ import annotations

from copy import copy
from enum import Enum
from importlib import import_module, invalidate_caches
import logging
import math
import os
import re
import signal
import sys

import pandas as pd

from .frameworks.definitions import load_framework_definition
from .job import Job, JobError, SimpleJobRunner, MultiThreadingJobRunner
from .datasets import DataLoader, DataSourceType
from .data import DatasetType
from .datautils import read_csv
from .resources import get as rget, config as rconfig, output_dirs as routput_dirs
from .results import ErrorResult, Scoreboard, TaskResult
from .utils import (
    Namespace as ns,
    OSMonitoring,
    as_list,
    datetime_iso,
    file_lock,
    flatten,
    json_dump,
    lazy_property,
    profile,
    repr_def,
    run_cmd,
    run_script,
    signal_handler,
    str2bool,
    str_sanitize,
    system_cores,
    system_memory_mb,
    system_volume_mb,
    touch,
)


log = logging.getLogger(__name__)

_setup_dir_ = ".setup"
_installed_file_ = "installed"
_setup_env_file_ = "setup_env"


class SetupMode(Enum):
    auto = 0
    skip = 1
    force = 2
    only = 3
    script = 4


class Benchmark:
    """Benchmark.
    Structure containing the generic information needed to run a benchmark:
     - the datasets
     - the automl framework


     we need to support:
     - openml tasks
     - openml datasets
     - openml studies (=benchmark suites)
     - user-defined (list of) datasets

     :param job_history: str or pd.DataFrame, default = None
        If specified, jobs will be skipped if their result is present in job_history.
        Useful to avoid duplicate work when trying to retry failed jobs.

    """

    data_loader = None
    framework_install_required = True

    def __init__(
        self,
        framework_name: str,
        benchmark_name: str,
        constraint_name: str,
        job_history: str | pd.DataFrame | None = None,
    ):
        self.job_runner = None

        if rconfig().run_mode == "script":
            # Used for recovery script
            self.framework_def, self.framework_name, self.framework_module = (
                None,
                None,
                None,
            )
            self.benchmark_def, self.benchmark_name, self.benchmark_path = (
                None,
                None,
                None,
            )
            self.constraint_def, self.constraint_name = None, None
            self.parallel_jobs = 1
            self.sid = None
            return

        self._forward_params = locals()
        if Benchmark.data_loader is None:
            Benchmark.data_loader = DataLoader(rconfig())

        self._job_history = self._load_job_history(job_history=job_history)
        framework = load_framework_definition(framework_name, rget())
        self.framework_def, self.framework_name = framework, framework.name
        log.debug("Using framework definition: %s.", self.framework_def)

        self.constraint_def, self.constraint_name = rget().constraint_definition(
            constraint_name
        )
        log.debug("Using constraint definition: %s.", self.constraint_def)

        self.benchmark_def, self.benchmark_name, self.benchmark_path = (
            rget().benchmark_definition(benchmark_name, self.constraint_def)
        )
        log.debug("Using benchmark definition: %s.", self.benchmark_def)

        self.parallel_jobs = rconfig().job_scheduler.parallel_jobs
        self.sid = (
            rconfig().sid
            if rconfig().sid is not None
            else rconfig()
            .token_separator.join(
                [
                    str_sanitize(framework_name),
                    str_sanitize(benchmark_name),
                    constraint_name,
                    rconfig().run_mode,
                    datetime_iso(micros=True, no_sep=True),
                ]
            )
            .lower()
        )

        self._validate()
        self.framework_module = import_module(self.framework_def.module)

    def _validate(self):
        if self.parallel_jobs > 1:
            log.warning(
                "Parallelization is not supported in local mode: ignoring `parallel_jobs=%s` parameter.",
                self.parallel_jobs,
            )
            self.parallel_jobs = 1

    def _load_job_history(self, job_history: str | pd.DataFrame | None) -> pd.DataFrame:
        """
        If job_history is None, return None
        If str, load result csv from str, return pandas DataFrame
        If pandas DataFrame, return pandas DataFrame
        """
        if job_history is None:
            return None
        if isinstance(job_history, str):
            log.info(f"Loading job history from {job_history}")
            job_history = read_csv(job_history)
        self._validate_job_history(job_history=job_history)
        return job_history

    def setup(self, mode: SetupMode):
        """
        ensure all dependencies needed by framework are available
        and possibly download them if necessary.
        Delegates specific setup to the framework module
        """
        if mode == SetupMode.skip or mode == SetupMode.auto and self._is_setup_done():
            return

        log.info("Setting up framework {}.".format(self.framework_name))

        self._write_setup_env(
            self.framework_module.__path__[0], **dict(self.framework_def.setup_env)
        )
        self._mark_setup_start()

        if hasattr(self.framework_module, "setup"):
            try:
                self.framework_module.setup(
                    *self.framework_def.setup_args,
                    _shell_=False,  # prevents #arg from being interpreted as comment
                    _live_output_=rconfig().setup.live_output,
                    _activity_timeout_=rconfig().setup.activity_timeout,
                )
            except Exception as e:
                raise JobError(
                    f"Setup of framework {self.framework_name} failed."
                ) from e

        if self.framework_def.setup_script is not None:
            run_script(
                self.framework_def.setup_script,
                _live_output_=rconfig().setup.live_output,
                _activity_timeout_=rconfig().setup.activity_timeout,
            )

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
            run_cmd(
                "\n".join(setup_cmd),
                _executable_="/bin/bash",
                _live_output_=rconfig().setup.live_output,
                _activity_timeout_=rconfig().setup.activity_timeout,
            )

        invalidate_caches()
        log.info(
            "Setup of framework {} completed successfully.".format(self.framework_name)
        )

        self._mark_setup_done()

    def _write_setup_env(self, dest_dir, **kwargs):
        setup_env = dict(AMLB_ROOT=rconfig().root_dir, PY_EXEC_PATH=sys.executable)
        setup_env.update(**kwargs)
        path = os.path.join(dest_dir, _setup_dir_, _setup_env_file_)
        touch(path)
        with open(path, "w") as f:
            f.write("\n".join([f"{k}={v}" for k, v in setup_env.items()] + [""]))

    def _installed_file(self):
        return os.path.join(self._framework_dir, _setup_dir_, _installed_file_)

    def _installed_version(self):
        installed = self._installed_file()
        versions = []
        if os.path.isfile(installed):
            with open(installed, "r") as f:
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
        if hasattr(self.framework_module, "version"):
            versions.append(self.framework_module.version())
        versions.extend([self.framework_def.version, ""])
        with open(installed, "a") as f:
            f.write("\n".join(versions))

    def cleanup(self):
        # anything to do?
        pass

    def run(
        self, tasks: str | list[str] | None = None, folds: int | list[int] | None = None
    ):
        """
        :param tasks: a single task name [str] or a list of task names to run. If None, then the whole benchmark will be used.
        :param folds: a fold [int] or a list of folds to run. If None, then the all folds from each task definition will be used.
        """
        try:
            assert (
                not self.framework_install_required or self._is_setup_done()
            ), f"Framework {self.framework_name} [{self.framework_def.version}] is not installed."

            task_defs = self._get_task_defs(tasks)
            jobs = flatten([self._task_jobs(task_def, folds) for task_def in task_defs])
            log.info(f"Running {len(jobs)} jobs")
            results = self._run_jobs(jobs)
            log.info(f"Processing results for {self.sid}")
            log.debug(results)

            if not rconfig().results.incremental_save:
                self._process_results(results)
            return self._results_summary()
        finally:
            self.cleanup()

    def _create_job_runner(self, jobs):
        on_new_result = (
            self._process_results if rconfig().results.incremental_save else None
        )
        if self.parallel_jobs == 1:
            return SimpleJobRunner(jobs, on_new_result=on_new_result)
        else:
            return MultiThreadingJobRunner(
                jobs,
                on_new_result=on_new_result,
                parallel_jobs=self.parallel_jobs,
                delay_secs=rconfig().job_scheduler.delay_between_jobs,
                done_async=True,
            )

    def _run_jobs(self, jobs):
        if not jobs:
            return []

        self.job_runner = self._create_job_runner(jobs)

        def on_interrupt(*_):
            log.warning("*** SESSION CANCELLED BY USER ***")
            log.warning(
                "*** Please wait for the application to terminate gracefully ***"
            )
            self.job_runner.stop()
            self.cleanup()
            # threading.Thread(target=self.job_runner.stop)
            # threading.Thread(target=self.cleanup)

        try:
            with signal_handler(signal.SIGINT, on_interrupt):
                with OSMonitoring(
                    name=jobs[0].name if len(jobs) == 1 else None,
                    interval_seconds=rconfig().monitoring.interval_seconds,
                    check_on_exit=True,
                    statistics=rconfig().monitoring.statistics,
                    verbosity=rconfig().monitoring.verbosity,
                ):
                    self.job_runner.start()
        except (KeyboardInterrupt, InterruptedError):
            pass
        finally:
            results = self.job_runner.results
        return results

    def _benchmark_tasks(self):
        return [
            task_def
            for task_def in self.benchmark_def
            if Benchmark._is_task_enabled(task_def)
        ]

    def _get_task_defs(self, task_name):
        task_defs = (
            self._benchmark_tasks()
            if task_name is None
            else [self._get_task_def(name) for name in task_name]
            if isinstance(task_name, list)
            else [self._get_task_def(task_name)]
        )
        if len(task_defs) == 0:
            raise ValueError("No task available.")
        return task_defs

    def _get_task_def(self, task_name, include_disabled=False, fail_on_missing=True):
        try:
            task_def = next(
                task
                for task in self.benchmark_def
                if task.name.lower() == str_sanitize(task_name.lower())
            )
        except StopIteration:
            if fail_on_missing:
                raise ValueError("Incorrect task name: {}.".format(task_name))
            return None
        if not include_disabled and not Benchmark._is_task_enabled(task_def):
            raise ValueError(
                f"Task {task_def.name} is disabled, please enable it first."
            )
        return task_def

    def _task_jobs(self, task_def, folds=None):
        folds = (
            range(task_def.folds)
            if folds is None
            else folds
            if isinstance(folds, list) and all(isinstance(f, int) for f in folds)
            else [folds]
            if isinstance(folds, int)
            else None
        )
        if folds is None:
            raise ValueError("Fold value should be None, an int, or a list of ints.")
        return list(filter(None, [self._make_job(task_def, f) for f in folds]))

    def _make_job(self, task_def, fold: int):
        """
        runs the framework against a given fold
        :param task_def: the task to run
        :param fold: the specific fold to use on this task
        """
        return (
            BenchmarkTask(self, task_def, fold).as_job()
            if not self._skip_job(task_def, fold)
            else None
        )

    def _in_job_history(self, task_def, fold):
        jh = self._job_history
        if jh is None:
            return False
        return (
            len(
                jh[
                    (jh.framework == self.framework_name)
                    & (jh.constraint == self.constraint_name)
                    & (jh.id == task_def.id)
                    & (jh.fold == fold)
                ]
            )
            > 0
        )

    @staticmethod
    def _validate_job_history(job_history):
        required_columns = {"framework", "constraint", "id", "fold"}
        actual_columns = set(job_history.columns)
        if missing_columns := (required_columns - actual_columns):
            quoted_columns = ", ".join(repr(c) for c in missing_columns)
            raise AssertionError(
                f"job_history missing required column(s) {quoted_columns}! "
            )

    def _skip_job(self, task_def, fold):
        if fold < 0 or fold >= task_def.folds:
            log.warning(
                f"Fold value {fold} is out of range for task {task_def.name}, skipping it."
            )
            return True

        if self._in_job_history(task_def, fold):
            log.info(
                f"Task {task_def.name} with fold {fold} is already present in job history, skipping it."
            )
            return True

        return False

    def _process_results(self, results):
        if not isinstance(results, list):
            results = [results]
        scores = list(filter(None, flatten([res.result for res in results])))
        if len(scores) == 0:
            return None

        for res in results:
            if math.isnan(res.result.duration):
                res.result.duration = res.duration

        board = Scoreboard(scores, scores_dir=self.output_dirs.scores)
        self._save(board)
        return board

    def _save(self, board):
        board.save(append=True)
        self._save_global(board)

    def _save_global(self, board):
        # Scoreboard.all().append(board).save()
        if rconfig().results.global_save:
            global_board = Scoreboard.all(rconfig().output_dir, autoload=False)
            dest_path = global_board.path
            timeout = rconfig().results.global_lock_timeout
            try:
                with file_lock(dest_path, timeout=timeout):
                    global_board.load().append(board).save()
            except TimeoutError:
                log.exception(
                    "Failed to acquire the lock on `%s` after %ss: "
                    "the partial board `%s` could not be appended to `%s`",
                    dest_path,
                    timeout,
                    board.path,
                    dest_path,
                )

    def _results_summary(self, scoreboard=None):
        board = scoreboard or Scoreboard.all(self.output_dirs.scores)
        results = board.as_printable_data_frame(verbosity=2)
        log.info(
            "Summing up scores for current run:\n%s",
            results.dropna(how="all", axis="columns").to_string(index=False),
        )
        return board.as_data_frame()

    @lazy_property
    def output_dirs(self):
        return routput_dirs(
            rconfig().output_dir,
            session=self.sid,
            subdirs=["predictions", "scores", "logs"],
        )

    @property
    def _framework_dir(self):
        return os.path.dirname(self.framework_module.__file__)

    @staticmethod
    def _is_task_enabled(task_def):
        return not hasattr(task_def, "enabled") or str2bool(str(task_def.enabled))


class TaskConfig:
    def __init__(
        self,
        name,
        openml_task_id,
        test_server,
        fold,
        metrics,
        quantile_levels,
        seed,
        max_runtime_seconds,
        cores,
        max_mem_size_mb,
        min_vol_size_mb,
        input_dir,
        output_dir,
        tag,
        command,
        git_info,
        measure_inference_time: bool = False,
    ):
        self.framework = None
        self.framework_params = None
        self.framework_version = None
        self.type = None
        self.name = name
        self.openml_task_id = openml_task_id
        self.test_server = test_server
        self.fold = fold
        self.metrics = [metrics] if isinstance(metrics, str) else metrics
        self.seed = seed
        self.max_runtime_seconds = max_runtime_seconds
        self.cores = cores
        self.max_mem_size_mb = max_mem_size_mb
        self.min_vol_size_mb = min_vol_size_mb
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_predictions_file = os.path.join(output_dir, "predictions.csv")
        self.tag = tag
        self.command = command
        self.git_info = git_info
        self.measure_inference_time = measure_inference_time
        self.ext = ns()  # used if frameworks require extra config points
        self.quantile_levels = list(sorted(quantile_levels))

    def __setattr__(self, name, value):
        if name == "metrics":
            self.metric = value[0] if isinstance(value, list) else value
        elif name == "max_runtime_seconds":
            inference_time_extension = 0
            if rconfig().inference_time_measurements.enabled:
                inference_time_extension = (
                    rconfig().inference_time_measurements.additional_job_time
                )
            overhead_time_multiplier = ns.get(
                rconfig(), "benchmarks.overhead_time_multiplier", 2
            )
            self.job_timeout_seconds = (
                min(
                    value * overhead_time_multiplier,
                    value + rconfig().benchmarks.overhead_time_seconds,
                )
                + inference_time_extension
            )
        super().__setattr__(name, value)

    def __json__(self):
        return self.__dict__

    def __repr__(self):
        return repr_def(self)

    def estimate_system_params(self):
        on_unfulfilled = rconfig().benchmarks.on_unfulfilled_constraint
        mode = re.split(r"\W+", rconfig().run_mode, maxsplit=1)[0]

        def handle_unfulfilled(message, on_auto="warn"):
            action = on_auto if on_unfulfilled == "auto" else on_unfulfilled
            if action == "warn":
                log.warning("WARNING: %s", message)
            elif action == "fail":
                raise JobError(message)

        sys_cores = system_cores()
        if self.cores > sys_cores:
            handle_unfulfilled(
                f"System with {sys_cores} cores does not meet requirements ({self.cores} cores)!.",
                on_auto="warn" if mode == "local" else "fail",
            )
        self.cores = min(self.cores, sys_cores) if self.cores > 0 else sys_cores
        log.info(
            "Assigning %s cores (total=%s) for new task %s.",
            self.cores,
            sys_cores,
            self.name,
        )

        sys_mem = system_memory_mb()
        os_recommended_mem = ns.get(
            rconfig(), f"{mode}.os_mem_size_mb", rconfig().benchmarks.os_mem_size_mb
        )
        left_for_app_mem = int(sys_mem.available - os_recommended_mem)
        assigned_mem = round(
            self.max_mem_size_mb
            if self.max_mem_size_mb > 0
            else left_for_app_mem
            if left_for_app_mem > 0
            else sys_mem.available
        )
        log.info(
            "Assigning %.f MB (total=%.f MB) for new %s task.",
            assigned_mem,
            sys_mem.total,
            self.name,
        )
        self.max_mem_size_mb = assigned_mem
        if assigned_mem > sys_mem.total:
            handle_unfulfilled(
                f"Total system memory {sys_mem.total} MB does not meet requirements ({assigned_mem} MB)!.",
                on_auto="fail",
            )
        elif assigned_mem > sys_mem.available:
            handle_unfulfilled(
                f"Assigned memory ({assigned_mem} MB) exceeds system available memory ({sys_mem.available} MB / total={sys_mem.total} MB)!"
            )
        elif assigned_mem > sys_mem.total - os_recommended_mem:
            handle_unfulfilled(
                f"Assigned memory ({assigned_mem} MB) is within {sys_mem.available} MB of system total memory {sys_mem.total} MB): "
                f"We recommend a {os_recommended_mem} MB buffer, otherwise OS memory usage might interfere with the benchmark task."
            )

        if self.min_vol_size_mb > 0:
            sys_vol = system_volume_mb()
            os_recommended_vol = rconfig().benchmarks.os_vol_size_mb
            if self.min_vol_size_mb > sys_vol.free:
                handle_unfulfilled(
                    f"Available storage ({sys_vol.free} MB / total={sys_vol.total} MB) does not meet requirements ({self.min_vol_size_mb+os_recommended_vol} MB)!"
                )


class BenchmarkTask:
    def __init__(self, benchmark: Benchmark, task_def, fold):
        """

        :param task_def:
        :param fold:
        """
        self.benchmark = benchmark
        self._task_def = task_def
        self.fold = fold
        self.task_config = TaskConfig(
            name=task_def.name,
            openml_task_id=task_def["openml_task_id"],
            fold=fold,
            metrics=task_def.metric,
            quantile_levels=task_def.quantile_levels,
            seed=rget().seed(fold),
            max_runtime_seconds=task_def.max_runtime_seconds,
            cores=task_def.cores,
            max_mem_size_mb=task_def.max_mem_size_mb,
            min_vol_size_mb=task_def.min_vol_size_mb,
            input_dir=rconfig().input_dir,
            output_dir=benchmark.output_dirs.session,
            test_server=rget().config.test_server,
            tag=rget().config.__dict__.get("tag"),
            command=rget().config.command,
            git_info=rget().git_info,
            measure_inference_time=rconfig().inference_time_measurements.enabled,
        )
        # allowing to override some task parameters through command line, e.g.: -Xt.max_runtime_seconds=60
        if rconfig()["t"] is not None:
            for c in dir(self.task_config):
                if rconfig().t[c] is not None:
                    setattr(self.task_config, c, rconfig().t[c])
        self._dataset = None

    def load_data(self):
        """
        Loads the training dataset for the current given task
        :return: path to the dataset file
        """
        if hasattr(self._task_def, "openml_task_id"):
            self._dataset = Benchmark.data_loader.load(
                DataSourceType.openml_task,
                task_id=self._task_def.openml_task_id,
                fold=self.fold,
            )
            log.debug(
                "Loaded OpenML dataset for task_id %s.", self._task_def.openml_task_id
            )
        elif hasattr(self._task_def, "openml_dataset_id"):
            raise NotImplementedError(
                "OpenML datasets without task_id are not supported yet."
            )
        elif hasattr(self._task_def, "dataset"):
            dataset_name_and_config = copy(self._task_def.dataset)
            dataset_name_and_config.name = self._task_def.name
            self._dataset = Benchmark.data_loader.load(
                DataSourceType.file, dataset=dataset_name_and_config, fold=self.fold
            )
        else:
            raise ValueError(
                "Tasks should have one property among [openml_task_id, openml_dataset_id, dataset]."
            )

    def as_job(self):
        job = Job(
            name=rconfig().token_separator.join(
                [
                    "local",
                    self.benchmark.benchmark_name,
                    self.benchmark.constraint_name,
                    self.task_config.name,
                    str(self.fold),
                    self.benchmark.framework_name,
                ]
            ),
            # specifying a job timeout to handle edge cases where framework never completes or hangs
            # (adding 5min safety to let the potential subprocess handle the interruption first).
            timeout_secs=self.task_config.job_timeout_seconds + 5 * 60,
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
        results = TaskResult(
            task_def=self._task_def,
            fold=self.fold,
            constraint=self.benchmark.constraint_name,
            predictions_dir=self.benchmark.output_dirs.predictions,
        )
        framework_def = self.benchmark.framework_def
        task_config = copy(self.task_config)
        if self._dataset.type == DatasetType.regression:
            task_config.type = "regression"
        elif self._dataset.type == DatasetType.timeseries:
            task_config.type = "timeseries"
        else:
            task_config.type = "classification"
        task_config.type_ = self._dataset.type.name
        task_config.framework = self.benchmark.framework_name
        task_config.framework_params = framework_def.params
        task_config.framework_version = self.benchmark._installed_version()[0]

        # allowing to pass framework parameters through command line, e.g.: -Xf.verbose=True -Xf.n_estimators=3000
        if rconfig()["f"] is not None:
            task_config.framework_params = ns.dict(
                ns(framework_def.params) + rconfig().f
            )

        task_config.output_predictions_file = results._predictions_file
        task_config.output_metadata_file = results._metadata_file
        touch(os.path.dirname(task_config.output_predictions_file), as_dir=True)
        if task_config.metrics is None:
            task_config.metrics = as_list(
                rconfig().benchmarks.metrics[self._dataset.type.name]
            )
            task_config.metric = task_config.metrics[0]

        result = meta_result = None
        try:
            log.info(
                "Running task %s on framework %s with config:\n%s",
                task_config.name,
                self.benchmark.framework_name,
                task_config,
            )
            json_dump(task_config, task_config.output_metadata_file, style="pretty")
            meta_result = self.benchmark.framework_module.run(
                self._dataset, task_config
            )
        except Exception as e:
            if rconfig().job_scheduler.exit_on_job_failure:
                raise
            log.exception(e)
            result = ErrorResult(e)
        finally:
            self._dataset.release()
        return results.compute_score(result=result, meta_result=meta_result)
