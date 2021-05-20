"""
**main** module handles all the main running logic:

- load specified framework and benchmark.
- extract the tasks and configure them.
- create jobs for each task.
- run the jobs.
- collect and save results.
"""
from enum import Enum
import logging
import math
import os
import re
import signal
from typing import List, Union

from .job import Job, JobError, SimpleJobRunner, MultiThreadingJobRunner
from .datasets import DataLoader
from .resources import get as rget, config as rconfig, output_dirs as routput_dirs
from .results import Scoreboard
from .utils import Namespace as ns, OSMonitoring, datetime_iso, flatten, lazy_property, \
    signal_handler, str2bool, str_sanitize, system_cores, system_memory_mb, system_volume_mb


log = logging.getLogger(__name__)


class SetupMode(Enum):
    auto = 0
    skip = 1
    force = 2
    only = 3
    script = 4


class Benchmark:

    run_mode = None
    data_loader = None

    def __init__(self, framework_name: str, benchmark_name: str, constraint_name: str):
        self.job_runner = None

        if not any([framework_name, benchmark_name, constraint_name]):
            # used to disable normal init in some unusual cases.
            self.framework_def, self.framework_name = None, None
            self.benchmark_def, self.benchmark_name, self.benchmark_path = None, None, None
            self.constraint_def, self.constraint_name = None, None
            self.parallel_jobs = 1
            self.sid = None
            return

        self._forward_params = locals()
        fsplits = framework_name.split(':', 1)
        framework_name = fsplits[0]
        tag = fsplits[1] if len(fsplits) > 1 else None
        self.framework_def, self.framework_name = rget().framework_definition(framework_name, tag)
        log.debug("Using framework definition: %s.", self.framework_def)

        self.constraint_def, self.constraint_name = rget().constraint_definition(constraint_name)
        log.debug("Using constraint definition: %s.", self.constraint_def)

        self.benchmark_def, self.benchmark_name, self.benchmark_path = rget().benchmark_definition(benchmark_name, self.constraint_def)
        log.debug("Using benchmark definition: %s.", self.benchmark_def)

        self.parallel_jobs = rconfig().job_scheduler.parallel_jobs
        self.sid = (rconfig().sid if rconfig().sid is not None
                    else rconfig().token_separator.join([
                        str_sanitize(framework_name),
                        str_sanitize(benchmark_name),
                        constraint_name,
                        self.run_mode,
                        datetime_iso(micros=True, no_sep=True)
                    ]).lower())

        self._validate()

    def _validate(self):
        pass

    @lazy_property
    def output_dirs(self):
        return routput_dirs(rconfig().output_dir, session=self.sid, subdirs=['predictions', 'scores', 'logs'])

    def setup(self, mode: SetupMode):
        self.data_loader = DataLoader(rconfig())

    def cleanup(self):
        pass

    def run(self, tasks: Union[str, List[str]] = None, folds: Union[int, List[int]] = None):
        """
        :param tasks: a single task name [str] or a list of task names to run. If None, then the whole benchmark will be used.
        :param folds: a fold [str] or a list of folds to run. If None, then the all folds from each task definition will be used.
        """
        jobs = self.create_jobs(tasks,  folds)
        return self.run_jobs(jobs)

    def create_jobs(self, tasks: Union[str, List[str]], folds: Union[int, List[int]]) -> List[Job]:
        task_defs = self._get_task_defs(tasks)
        jobs = flatten([self._task_jobs(task_def, folds) for task_def in task_defs])
        return jobs

    def run_jobs(self, jobs):
        try:
            results = self._run_jobs(jobs)
            log.info(f"Processing results for {self.sid}")
            log.debug(results)
            return self._process_results(results)
        finally:
            self.cleanup()

    def _get_task_defs(self, task_name):
        task_defs = (self._benchmark_tasks() if task_name is None
                     else [self._get_task_def(name) for name in task_name] if isinstance(task_name, list)
        else [self._get_task_def(task_name)])
        if len(task_defs) == 0:
            raise ValueError("No task available.")
        return task_defs

    def _benchmark_tasks(self):
        return [task_def for task_def in self.benchmark_def if self._is_task_enabled(task_def)]

    def _get_task_def(self, task_name, include_disabled=False, fail_on_missing=True):
        try:
            task_def = next(task for task in self.benchmark_def if task.name.lower() == str_sanitize(task_name.lower()))
        except StopIteration:
            if fail_on_missing:
                raise ValueError("Incorrect task name: {}.".format(task_name))
            return None
        if not include_disabled and not self._is_task_enabled(task_def):
            raise ValueError(f"Task {task_def.name} is disabled, please enable it first.")
        return task_def

    def _task_jobs(self, task_def, folds=None):
        folds = (range(task_def.folds) if folds is None
                 else folds if isinstance(folds, list) and all(isinstance(f, int) for f in folds)
        else [folds] if isinstance(folds, int)
        else None)
        if folds is None:
            raise ValueError("Fold value should be None, an int, or a list of ints.")
        return list(filter(None, [self._make_job(task_def, f) for f in folds]))

    def _make_job(self, task_def, fold: int):
        pass

    def _run_jobs(self, jobs: List[Job]):
        self.job_runner = self._create_job_runner(jobs)

        def on_interrupt(*_):
            log.warning("*** SESSION CANCELLED BY USER ***")
            log.warning("*** Please wait for the application to terminate gracefully ***")
            self.job_runner.stop()
            self.cleanup()
            # threading.Thread(target=self.job_runner.stop)
            # threading.Thread(target=self.cleanup)

        try:
            with signal_handler(signal.SIGINT, on_interrupt):
                with OSMonitoring(name=jobs[0].name if len(jobs) == 1 else None,
                                  interval_seconds=rconfig().monitoring.interval_seconds,
                                  check_on_exit=True,
                                  statistics=rconfig().monitoring.statistics,
                                  verbosity=rconfig().monitoring.verbosity):
                    self.job_runner.start()
        except (KeyboardInterrupt, InterruptedError):
            pass
        finally:
            results = self.job_runner.results

        for res in results:
            if res.result is not None and math.isnan(res.result.duration):
                res.result.duration = res.duration
        return results

    def _create_job_runner(self, jobs):
        if self.parallel_jobs == 1:
            return SimpleJobRunner(jobs)
        else:
            # return ThreadPoolExecutorJobRunner(jobs, self.parallel_jobs)
            return MultiThreadingJobRunner(jobs, self.parallel_jobs,
                                           delay_secs=rconfig().job_scheduler.delay_between_jobs,
                                           done_async=True)

    def _process_results(self, results):
        scores = list(filter(None, flatten([res.result for res in results])))
        if len(scores) == 0:
            return None

        board = Scoreboard(scores,
                           framework_name=self.framework_name,
                           benchmark_name=self.benchmark_name,
                           scores_dir=self.output_dirs.scores)
        self._save(board)

        log.info("Summing up scores for current run:\n%s",
                 board.as_printable_data_frame(verbosity=2).dropna(how='all', axis='columns').to_string(index=False))
        return board.as_data_frame()

    def _save(self, board):
        board.save(append=True)
        self._append(board)

    def _append(self, board):
        Scoreboard.all().append(board).save()
        Scoreboard.all(rconfig().output_dir).append(board).save()

    @classmethod
    def _is_task_enabled(cls, task_def):
        return not hasattr(task_def, 'enabled') or str2bool(str(task_def.enabled))


class TaskConfig:

    def __init__(self, name, fold, metrics, seed,
                 max_runtime_seconds, cores, max_mem_size_mb, min_vol_size_mb,
                 input_dir, output_dir, run_mode):
        self.framework = None
        self.framework_params = None
        self.framework_version = None
        self.type = None
        self.name = name
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
        self.run_mode = run_mode
        self.ext = ns()  # used if frameworks require extra config points

    def __setattr__(self, name, value):
        if name == 'metrics':
            self.metric = value[0] if isinstance(value, list) else value
        elif name == 'max_runtime_seconds':
            self.job_timeout_seconds = min(value * 2,
                                           value + rconfig().benchmarks.overhead_time_seconds)
        super().__setattr__(name, value)

    def __json__(self):
        return self.__dict__

    def estimate_system_params(self):
        on_unfulfilled = rconfig().benchmarks.on_unfulfilled_constraint
        mode = re.split(r"\W+", rconfig().run_mode, maxsplit=1)[0]

        def handle_unfulfilled(message, on_auto='warn'):
            action = on_auto if on_unfulfilled == 'auto' else on_unfulfilled
            if action == 'warn':
                log.warning("WARNING: %s", message)
            elif action == 'fail':
                raise JobError(message)

        sys_cores = system_cores()
        if self.cores > sys_cores:
            handle_unfulfilled(f"System with {sys_cores} cores does not meet requirements ({self.cores} cores)!.",
                               on_auto='warn' if mode == 'local' else 'fail')
        self.cores = min(self.cores, sys_cores) if self.cores > 0 else sys_cores
        log.info("Assigning %s cores (total=%s) for new task %s.", self.cores, sys_cores, self.name)

        sys_mem = system_memory_mb()
        os_recommended_mem = ns.get(rconfig(), f"{mode}.os_mem_size_mb", rconfig().benchmarks.os_mem_size_mb)
        left_for_app_mem = int(sys_mem.available - os_recommended_mem)
        assigned_mem = round(self.max_mem_size_mb if self.max_mem_size_mb > 0
                             else left_for_app_mem if left_for_app_mem > 0
                             else sys_mem.available)
        log.info("Assigning %.f MB (total=%.f MB) for new %s task.", assigned_mem, sys_mem.total, self.name)
        self.max_mem_size_mb = assigned_mem
        if assigned_mem > sys_mem.total:
            handle_unfulfilled(f"Total system memory {sys_mem.total} MB does not meet requirements ({assigned_mem} MB)!.",
                               on_auto='fail')
        elif assigned_mem > sys_mem.available:
            handle_unfulfilled(f"Assigned memory ({assigned_mem} MB) exceeds system available memory ({sys_mem.available} MB / total={sys_mem.total} MB)!")
        elif assigned_mem > sys_mem.total - os_recommended_mem:
            handle_unfulfilled(f"Assigned memory ({assigned_mem} MB) is within {sys_mem.available} MB of system total memory {sys_mem.total} MB): "
                               f"We recommend a {os_recommended_mem} MB buffer, otherwise OS memory usage might interfere with the benchmark task.")

        if self.min_vol_size_mb > 0:
            sys_vol = system_volume_mb()
            os_recommended_vol = rconfig().benchmarks.os_vol_size_mb
            if self.min_vol_size_mb > sys_vol.free:
                handle_unfulfilled(f"Available storage ({sys_vol.free} MB / total={sys_vol.total} MB) does not meet requirements ({self.min_vol_size_mb+os_recommended_vol} MB)!")


