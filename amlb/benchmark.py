"""
**benchmark** module handles all the main logic:

- load specified framework and benchmark.
- extract the tasks and configure them.
- create jobs for each task.
- run the jobs.
- collect and save results.
"""

from copy import copy
from enum import Enum
from importlib import import_module, invalidate_caches
import logging
import math
import os

from .job import Job, SimpleJobRunner, MultiThreadingJobRunner, ThreadPoolExecutorJobRunner, ProcessPoolExecutorJobRunner
from .datasets import DataLoader, DataSourceType
from .data import DatasetType
from .resources import get as rget, config as rconfig, output_dirs as routput_dirs
from .results import ErrorResult, Scoreboard, TaskResult
from .utils import Namespace as ns, OSMonitoring, as_list, datetime_iso, flatten, lazy_property, profile, repr_def, \
    run_cmd, run_script, str2bool, system_cores, system_memory_mb, system_volume_mb, touch


log = logging.getLogger(__name__)


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
    """

    data_loader = None

    def __init__(self, framework_name: str, benchmark_name: str, constraint_name: str):
        if rconfig().run_mode == 'script':
            self.framework_def, self.framework_name, self.framework_module = None, None, None
            self.benchmark_def, self.benchmark_name, self.benchmark_path = None, None, None
            self.constraint_def, self.constraint_name = None, None
            self.parallel_jobs = 1
            self.sid = None
            return

        self.framework_def, self.framework_name = rget().framework_definition(framework_name)
        log.debug("Using framework definition: %s.", self.framework_def)

        self.constraint_def, self.constraint_name = rget().constraint_definition(constraint_name)
        log.debug("Using constraint definition: %s.", self.constraint_def)

        self.benchmark_def, self.benchmark_name, self.benchmark_path = rget().benchmark_definition(benchmark_name, self.constraint_def)
        log.debug("Using benchmark definition: %s.", self.benchmark_def)

        self.parallel_jobs = rconfig().parallel_jobs
        self.sid = (rconfig().sid if rconfig().sid is not None
                    else rconfig().token_separator.join([
                        rconfig().token_separator.join([
                            framework_name,
                            benchmark_name,
                            constraint_name,
                            rconfig().run_mode
                        ]).lower(),
                        datetime_iso(micros=True, no_sep=True)
                    ])).replace("/", "_")

        self._validate()
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
        Benchmark.data_loader = DataLoader(rconfig())

        if mode == SetupMode.skip or mode == SetupMode.auto and self._setup_done():
            return

        log.info("Setting up framework {}.".format(self.framework_name))

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

        self._setup_done(mark=True)

    def cleanup(self):
        # anything to do?
        pass

    def run(self, task_name=None, fold=None):
        """
        :param task_name: a single task name [str] or a list of task names to run. If None, then the whole benchmark will be used.
        :param fold: a fold [str] or a list of folds to run. If None, then the all folds from each task definition will be used.
        """
        task_defs = self._get_task_defs(task_name)
        jobs = flatten([self._task_jobs(task_def, fold) for task_def in task_defs])
        try:
            results = self._run_jobs(jobs)
            log.info(f"Processing results for {self.sid}")
            log.debug(results)
            if task_name is None:
                scoreboard = self._process_results(results)
            else:
                for task_def in task_defs:
                    task_results = filter(lambda res: res.result is not None and res.result.task == task_def.name, results)
                    scoreboard = self._process_results(task_results, task_name=task_def.name)
            return scoreboard
        finally:
            self.cleanup()

    def _run_jobs(self, jobs):
        if self.parallel_jobs == 1:
            runner = SimpleJobRunner(jobs)
        else:
            # runner = ThreadPoolExecutorJobRunner(jobs, self.parallel_jobs)
            runner = MultiThreadingJobRunner(jobs, self.parallel_jobs, delay_secs=rconfig().delay_between_jobs, done_async=True)

        try:
            with OSMonitoring(name=jobs[0].name if len(jobs) == 1 else None,
                              frequency_seconds=rconfig().monitoring.frequency_seconds,
                              check_on_exit=True,
                              statistics=rconfig().monitoring.statistics,
                              verbosity=rconfig().monitoring.verbosity):
                runner.start()
        except (KeyboardInterrupt, InterruptedError):
            pass
        finally:
            results = runner.results

        for res in results:
            if res.result is not None and math.isnan(res.result.duration):
                res.result.duration = res.duration
        return results

    def _benchmark_tasks(self):
        return [task_def for task_def in self.benchmark_def if Benchmark._is_task_enabled(task_def)]

    def _get_task_defs(self, task_name):
        task_defs = self._benchmark_tasks() if task_name is None \
            else [self._get_task_def(name) for name in task_name] if isinstance(task_name, list) \
            else [self._get_task_def(task_name)]
        if len(task_defs) == 0:
            raise ValueError("No task available.")
        return task_defs

    def _get_task_def(self, task_name, include_disabled=False, fail_on_missing=True):
        try:
            task_def = next(task for task in self.benchmark_def if task.name.lower() == task_name.lower())
        except StopIteration:
            if fail_on_missing:
                raise ValueError("Incorrect task name: {}.".format(task_name))
            return None
        if not include_disabled and not Benchmark._is_task_enabled(task_def):
            raise ValueError(f"Task {task_def.name} is disabled, please enable it first.")
        return task_def

    def _task_jobs(self, task_def, folds=None):
        folds = range(task_def.folds) if folds is None \
            else folds if isinstance(folds, list) and all(isinstance(f, int) for f in folds) \
            else [folds] if isinstance(folds, int) \
            else None
        if folds is None:
            raise ValueError("Fold value should be None, an int, or a list of ints.")
        return list(filter(None, [self._make_job(task_def, f) for f in folds]))

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

        return BenchmarkTask(self, task_def, fold).as_job(self.framework_module, self.framework_name)

    def _process_results(self, results, task_name=None):
        scores = list(filter(None, flatten([res.result for res in results])))
        if len(scores) == 0:
            return None

        board = Scoreboard(scores,
                           framework_name=self.framework_name,
                           task_name=task_name,
                           scores_dir=self.output_dirs.scores) if task_name \
            else Scoreboard(scores,
                            framework_name=self.framework_name,
                            benchmark_name=self.benchmark_name,
                            scores_dir=self.output_dirs.scores)

        if rconfig().results.save:
            self._save(board)

        log.info("Summing up scores for current run:\n%s", board.as_printable_data_frame().dropna(how='all', axis='columns').to_string())
        return board.as_data_frame()

    def _save(self, board):
        board.save(append=True)
        self._append(board)

    def _append(self, board):
        Scoreboard.all().append(board).save()
        Scoreboard.all(rconfig().output_dir).append(board).save()

    def _setup_done(self, mark=False):
        marker_file = os.path.join(self._framework_dir, '.marker_setup_safe_to_delete')
        setup_done = os.path.isfile(marker_file)
        if mark and not setup_done:
            touch(marker_file)
            setup_done = True
        return setup_done

    @lazy_property
    def output_dirs(self):
        return routput_dirs(rconfig().output_dir, session=self.sid, subdirs=['predictions', 'scores', 'logs'])

    @property
    def _framework_dir(self):
        return os.path.dirname(self.framework_module.__file__)

    @staticmethod
    def _is_task_enabled(task_def):
        return not hasattr(task_def, 'enabled') or str2bool(str(task_def.enabled))


class TaskConfig:

    def __init__(self, name, fold, metrics, seed,
                 max_runtime_seconds, cores, max_mem_size_mb, min_vol_size_mb,
                 input_dir, output_dir):
        self.framework = None
        self.framework_params = None
        self.type = None
        self.name = name
        self.fold = fold
        self.metrics = [metrics] if isinstance(metrics, str) else metrics
        self.metric = metrics[0] if isinstance(metrics, list) else metrics
        self.seed = seed
        self.max_runtime_seconds = max_runtime_seconds
        self.cores = cores
        self.max_mem_size_mb = max_mem_size_mb
        self.min_vol_size_mb = min_vol_size_mb
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_predictions_file = os.path.join(output_dir, "predictions.csv")

    def __json__(self):
        return self.__dict__

    def estimate_system_params(self):
        sys_cores = system_cores()
        self.cores = min(self.cores, sys_cores) if self.cores > 0 else sys_cores
        log.info("Assigning %s cores (total=%s) for new task %s.", self.cores, sys_cores, self.name)

        sys_mem = system_memory_mb()
        os_recommended_mem = rconfig().benchmarks.os_mem_size_mb
        left_for_app_mem = int(sys_mem.available - os_recommended_mem)
        assigned_mem = round(self.max_mem_size_mb if self.max_mem_size_mb > 0
                             else left_for_app_mem if left_for_app_mem > 0
                             else sys_mem.available)
        log.info("Assigning %.f MB (total=%.f MB) for new %s task.", assigned_mem, sys_mem.total, self.name)
        self.max_mem_size_mb = assigned_mem
        if assigned_mem > sys_mem.available:
            log.warning("WARNING: Assigned memory (%(assigned).f MB) exceeds system available memory (%(available).f MB / total=%(total).f MB)!",
                        dict(assigned=assigned_mem, available=sys_mem.available, total=sys_mem.total))
        elif assigned_mem > sys_mem.total - os_recommended_mem:
            log.warning("WARNING: Assigned memory (%(assigned).f MB) is within %(buffer).f MB of system total memory (%(total).f MB): "
                        "We recommend a %(buffer).f MB buffer, otherwise OS memory usage might interfere with the benchmark task.",
                        dict(assigned=assigned_mem, available=sys_mem.available, total=sys_mem.total, buffer=os_recommended_mem))

        if self.min_vol_size_mb > 0:
            sys_vol = system_volume_mb()
            os_recommended_vol = rconfig().benchmarks.os_vol_size_mb
            if self.min_vol_size_mb > sys_vol.free:
                log.warning("WARNING: Available volume memory (%(available).f MB / total=%(total).f MB) doesn't meet requirements (%(required).f MB)!",
                            dict(required=self.min_vol_size_mb+os_recommended_vol, available=sys_vol.free, total=sys_vol.total))


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
            fold=fold,
            metrics=task_def.metric,
            seed=rget().seed(fold),
            max_runtime_seconds=task_def.max_runtime_seconds,
            cores=task_def.cores,
            max_mem_size_mb=task_def.max_mem_size_mb,
            min_vol_size_mb=task_def.min_vol_size_mb,
            input_dir=rconfig().input_dir,
            output_dir=benchmark.output_dirs.session,
        )
        # allowing to override some task parameters through command line, e.g.: -Xt.max_runtime_seconds=60
        if rconfig()['t'] is not None:
            for c in ['max_runtime_seconds', 'metric', 'metrics']:
                if rconfig().t[c] is not None:
                    setattr(self.task_config, c, rconfig().t[c])
        self._dataset = None

    @profile(logger=log)
    def load_data(self):
        """
        Loads the training dataset for the current given task
        :return: path to the dataset file
        """
        if hasattr(self._task_def, 'openml_task_id'):
            self._dataset = Benchmark.data_loader.load(DataSourceType.openml_task, task_id=self._task_def.openml_task_id, fold=self.fold)
            log.debug("Loaded OpenML dataset for task_id %s.", self._task_def.openml_task_id)
        elif hasattr(self._task_def, 'openml_dataset_id'):
            # TODO
            raise NotImplementedError("OpenML datasets without task_id are not supported yet.")
        elif hasattr(self._task_def, 'dataset'):
            self._dataset = Benchmark.data_loader.load(DataSourceType.file, dataset=self._task_def.dataset, fold=self.fold)
        else:
            raise ValueError("Tasks should have one property among [openml_task_id, openml_dataset_id, dataset].")

    def as_job(self, framework, framework_name):
        def _run():
            self.load_data()
            return self.run(framework, framework_name)
        timeout_secs = min(self.task_config.max_runtime_seconds * 2,
                           self.task_config.max_runtime_seconds + rconfig().benchmarks.overhead_time_seconds)
        job = Job(name=rconfig().token_separator.join(['local', self.task_config.name, str(self.fold), framework_name]),
                  timeout_secs=timeout_secs)  # this timeout is just to handle edge cases where framework never completes
        job._run = _run
        return job
        # return Namespace(run=lambda: self.run(framework))

    @profile(logger=log)
    def run(self, framework, framework_name):
        """

        :param framework:
        :return:
        """
        results = TaskResult(task_def=self._task_def, fold=self.fold,
                             constraint=self.benchmark.constraint_name,
                             predictions_dir=self.benchmark.output_dirs.predictions)
        framework_def, _ = rget().framework_definition(framework_name)
        task_config = copy(self.task_config)
        task_config.estimate_system_params()
        task_config.type = 'regression' if self._dataset.type == DatasetType.regression else 'classification'
        task_config.framework = framework_name
        task_config.framework_params = framework_def.params

        # allowing to pass framework parameters through command line, e.g.: -Xf.verbose=True -Xf.n_estimators=3000
        if rconfig()['f'] is not None:
            task_config.framework_params = ns.dict(ns(framework_def.params) + rconfig().f)

        task_config.output_predictions_file = results._predictions_file(task_config.framework.lower())
        touch(os.path.dirname(task_config.output_predictions_file), as_dir=True)
        if task_config.metrics is None:
            task_config.metrics = as_list(rconfig().benchmarks.metrics[self._dataset.type.name])
            task_config.metric = task_config.metrics[0]

        result = meta_result = None
        try:
            log.info("Running task %s on framework %s with config:\n%s", task_config.name, framework_name, repr_def(task_config))
            meta_result = framework.run(self._dataset, task_config)
        except Exception as e:
            log.exception(e)
            result = ErrorResult(e)
        finally:
            self._dataset.release()

        meta_result = meta_result or {}
        meta_result['params'] = task_config.framework_params
        return results.compute_scores(framework_name, task_config.metrics, result=result, meta_result=meta_result)

