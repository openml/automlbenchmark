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
from importlib import import_module
import logging
import os
import queue
import random as rnd
import threading
import time

from .openml import Openml
from .resources import get as rget, config as rconfig
from .results import Scoreboard, TaskResult
from .utils import Namespace, datetime_iso, flatten, profile, run_cmd, str2bool, system_cores, system_memory_mb, touch as ftouch


log = logging.getLogger(__name__)


class Benchmark:
    """Benchmark.
    Structure containing the generic information needed to run a benchmark:
     - the datasets
     - the automl framework


     we need to support:
     - openml datasets
     - openml tasks
     - openml studies (=benchmark suites)
     - user-defined (list of) datasets
    """

    task_loader = None
    SetupMode = Enum('SetupMode', 'auto skip force only')

    def __init__(self, framework_name: str, benchmark_name: str, parallel_jobs=1):
        """

        :param framework_name:
        :param benchmark_name:
        :param resources:
        """
        self.framework_def, self.framework_name = rget().framework_definition(framework_name)
        log.debug("Using framework definition: %s", self.framework_def)
        self.benchmark_def, self.benchmark_name, self.benchmark_path = rget().benchmark_definition(benchmark_name)
        log.debug("Using benchmark definition: %s", self.benchmark_def)
        self.parallel_jobs = parallel_jobs
        self.uid = "{}-{}-{}".format(framework_name, benchmark_name, datetime_iso(micros=True, no_sep=True)).lower()

        self.framework_module = import_module(self.framework_def.module)

    def _validate(self):
        if self.parallel_jobs > 1:
            log.warning("parallelization is not supported in local mode: ignoring `parallel_jobs` parameter")
            self.parallel_jobs = 1

    def setup(self, mode: SetupMode):
        """
        ensure all dependencies needed by framework are available
        and possibly download them if necessary.
        Delegates specific setup to the framework module
        """
        Benchmark.task_loader = Openml(api_key=rconfig().openml.apikey, cache_dir=rconfig().input_dir)
        if mode == Benchmark.SetupMode.skip or not hasattr(self.framework_module, 'setup'):
            return

        if mode == Benchmark.SetupMode.auto and self._setup_done():
            return

        log.info("Setting up framework {}.".format(self.framework_name))
        self.framework_module.setup(self.framework_def.setup_args)
        if self.framework_def.setup_cmd is not None:
            output = run_cmd(self.framework_def.setup_cmd)
            log.debug(output)
        log.info("Setup of framework {} completed successfully.".format(self.framework_name))

        self._setup_done(touch=True)

    def cleanup(self):
        # anything to do?
        pass

    def run(self, save_scores=False):
        """
        runs the framework for every task in the benchmark definition
        """
        results = self._run_jobs(self._benchmark_jobs())
        return self._process_results(results, save_scores=save_scores)

    def run_one(self, task_name: str, fold, save_scores=False):
        """

        :param task_name:
        :param fold:
        :param save_scores:
        """
        task_def = self._get_task_def(task_name)
        results = self._run_jobs(self._custom_task_jobs(task_def, fold))
        return self._process_results(results, task_name=task_name, save_scores=save_scores)

    def _run_jobs(self, jobs):
        if self.parallel_jobs == 1:
            return SimpleJobRunner(jobs).start()
        else:
            return ParallelJobRunner(jobs, self.parallel_jobs).start()

    def _benchmark_jobs(self):
        jobs = []
        for task_def in self.benchmark_def:
            if Benchmark._is_task_enabled(task_def):
                jobs.extend(self._task_jobs(task_def))
        return jobs

    def _custom_task_jobs(self, task_def, folds):
        jobs = []
        if folds is None:
            jobs.extend(self._task_jobs(task_def))
        elif isinstance(folds, int):
            jobs.append(self._fold_job(task_def, folds))
        elif isinstance(folds, list) and all(isinstance(f, int) for f in folds):
            for f in folds:
                jobs.append(self._fold_job(task_def, f))
        else:
            raise ValueError("fold value should be None, an int, or a list of ints")
        return jobs

    def _task_jobs(self, task_def):
        """
        run the framework for every fold in the task definition
        :param task_def:
        """
        jobs = []
        for fold in range(task_def.folds):
            jobs.append(self._fold_job(task_def, fold))
        return jobs

    def _fold_job(self, task_def, fold: int):
        """
        runs the framework against a given fold
        :param task_def: the task to run
        :param fold: the specific fold to use on this task
        """
        if fold < 0 or fold >= task_def.folds:
            raise ValueError("fold value {} is out of range for task {}".format(fold, task_def.name))

        return BenchmarkTask(task_def, fold).as_job(self.framework_module, self.framework_name)

    def _get_task_def(self, task_name):
        try:
            task_def = next(task for task in self.benchmark_def if task.name == task_name)
        except StopIteration:
            raise ValueError("incorrect task name: {}".format(task_name))
        if not Benchmark._is_task_enabled(task_def):
            raise ValueError("task {} is disabled, please enable it first".format(task_name))
        return task_def

    def _process_results(self, results, task_name=None, save_scores=False):
        scores = flatten([res.result for res in results])
        if len(scores) == 0 or not any(scores):
            return None

        board = Scoreboard(scores, framework_name=self.framework_name, task_name=task_name) if task_name \
            else Scoreboard(scores, framework_name=self.framework_name, benchmark_name=self.benchmark_name)

        if save_scores:
            self._save(board)

        log.info("Summing up scores for current run:\n%s", board.as_data_frame())
        return board.as_data_frame()

    def _save(self, board):
        board.save(append=True)
        Scoreboard.all().append(board).save()

    def _setup_done(self, touch=False):
        marker_file = os.path.join(self._framework_dir, '.marker_setup_safe_to_delete')
        setup_done = os.path.isfile(marker_file)
        if touch and not setup_done:
            ftouch(marker_file)
            setup_done = True
        return setup_done

    @property
    def _framework_dir(self):
        return os.path.dirname(self.framework_module.__file__)

    @staticmethod
    def _is_task_enabled(task_def):
        return not hasattr(task_def, 'enabled') or str2bool(str(task_def.enabled))


class TaskConfig:

    @staticmethod
    def from_def(task_def, fold, config):
        return TaskConfig(
            name=task_def.name,
            fold=fold,
            metrics=task_def.metric,
            max_runtime_seconds=task_def.max_runtime_seconds,
            cores=task_def.cores,
            max_mem_size_mb=task_def.max_mem_size_mb,
            input_dir=config.input_dir,
            output_dir=config.predictions_dir,
        )

    def __init__(self, name, fold, metrics, max_runtime_seconds,
                 cores, max_mem_size_mb,
                 input_dir, output_dir):
        self.framework = None
        self.name = name
        self.fold = fold
        self.metrics = [metrics] if isinstance(metrics, str) else metrics
        self.metric = metrics[0] if isinstance(metrics, list) else metrics
        self.max_runtime_seconds = max_runtime_seconds
        self.cores = cores
        self.max_mem_size_mb = max_mem_size_mb
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_predictions_file = os.path.join(output_dir, "predictions.csv")

    def estimate_system_params(self):
        sys_cores = system_cores()
        self.cores = min(self.cores, sys_cores) if self.cores > 0 else sys_cores
        log.info("Assigning %s cores (total=%s) for new task %s.", self.cores, sys_cores, self.name)

        sys_mem = system_memory_mb()
        os_recommended_mem = rconfig().benchmarks.os_mem_size_mb
        # os is already using mem, so leaving half of recommended mem
        assigned_mem = round(self.max_mem_size_mb if self.max_mem_size_mb > 0 else int(sys_mem.available - os_recommended_mem / 2))
        log.info("Assigning %sMB (total=%sMB) for new %s task.", assigned_mem, sys_mem.total, self.name)
        self.max_mem_size_mb = assigned_mem
        if assigned_mem > sys_mem.available:
            log.warning("BEWARE! Assigned memory (%(assigned)sMB) exceeds system available memory (%(available)sMB / total=%(total)sMB)!",
                        dict(assigned=assigned_mem, available=sys_mem.available, total=sys_mem.total))
        elif assigned_mem > sys_mem.total - os_recommended_mem:
            log.warning("BEWARE! Assigned memory (%(assigned)sMB) is within %(buffer)sMB of system total memory (%(total)sMB): "
                        "We recommend a %(buffer)sMB buffer, otherwise OS memory usage might interfere with the benchmark task.",
                        dict(assigned=assigned_mem, available=sys_mem.available, total=sys_mem.total, buffer=os_recommended_mem))


class BenchmarkTask:
    """

    """

    def __init__(self, task_def, fold):
        """

        :param task_def:
        :param fold:
        """
        self._task_def = task_def
        self.fold = fold
        self.task = TaskConfig.from_def(self._task_def, self.fold, rconfig())
        self._dataset = None

    @profile(logger=log)
    def load_data(self):
        """
        Loads the training dataset for the current given task
        :return: path to the dataset file
        """
        if hasattr(self._task_def, 'openml_task_id'):
            self._dataset = Benchmark.task_loader.load(task_id=self._task_def.openml_task_id, fold=self.fold)
            log.debug("Loaded OpenML dataset for task_id %s", self._task_def.openml_task_id)
        elif hasattr(self._task_def, 'dataset'):
            # todo
            raise NotImplementedError("Raw dataset are not supported yet")
        else:
            raise ValueError("Tasks should have one property amtaskong [openml_task_id, dataset]")

    def as_job(self, framework, framework_name):
        def _run():
            self.load_data()
            return self.run(framework, framework_name)
        job = Job("local_{}_{}_{}".format(self.task.name, self.fold, framework_name))
        job._run = _run
        return job
        # return Namespace(run=lambda: self.run(framework))

    @profile(logger=log)
    def run(self, framework, framework_name):
        """

        :param framework:
        :return:
        """
        results = TaskResult(task_name=self.task.name, fold=self.fold)
        task_config = copy(self.task)
        task_config.framework = framework_name
        task_config.output_predictions_file = results._predictions_file(task_config.framework.lower())
        task_config.estimate_system_params()
        framework.run(self._dataset, task_config)
        self._dataset.release()
        return results.compute_scores(framework_name, task_config.metrics)


class Job:

    def __init__(self, name=""):
        self.name = name

    def run(self):
        try:
            beep = time.time()
            result = self._run()
            duration = time.time() - beep
            log.info("Job %s executed in %.3f seconds", self.name, duration)
            log.debug("Job %s returned: %s", self.name, result)
            return result, duration
        except Exception as e:
            log.error("Job `%s` failed with error %s", self.name, str(e))
            log.exception(e)
            return None, -1

    def done(self):
        try:
            self._on_done()
        except Exception as e:
            log.error("Job `%s` completion failed with error %s", self.name, str(e))
            log.exception(e)

    def _run(self):
        """jobs should implement their run logic in this method"""
        pass

    def _on_done(self):
        """hook to execute logic after job completion in a thread-safe way as this is executed in the main thread"""
        pass


class JobRunner:

    State = Enum('State', 'created running stopping stopped')

    def __init__(self, jobs):
        self.jobs = jobs
        self.results = []
        self.state = JobRunner.State.created

    def start(self):
        start_time = time.time()
        self.state = JobRunner.State.running
        self._run()
        self.state = JobRunner.State.stopped
        stop_time = time.time()
        log.info("All jobs executed in %.3f seconds", stop_time-start_time)
        return self.results

    def stop(self):
        self.state = JobRunner.State.stopping

    def _run(self):
        pass


class SimpleJobRunner(JobRunner):

    def _run(self):
        for job in self.jobs:
            if self.state == JobRunner.State.stopping:
                break
            result, duration = job.run()
            self.results.append(Namespace(name=job.name, result=result, duration=duration))
            job.done()


class ParallelJobRunner(JobRunner):

    def __init__(self, jobs, parallel_jobs, done_async=False):
        super().__init__(jobs)
        self.parallel_jobs = parallel_jobs
        self.done_async = done_async

    def _run(self):
        q = queue.Queue()

        def worker():
            while True:
                job = q.get()
                if job is None or self.state == JobRunner.State.stopping:
                    break
                result, duration = job.run()
                self.results.append(Namespace(name=job.name, result=result, duration=duration))
                if self.done_async:
                    job.done()
                q.task_done()

        threads = []
        for thread in range(self.parallel_jobs):
            thread = threading.Thread(target=worker)
            thread.start()
            threads.append(thread)

        try:
            for job in self.jobs:
                q.put(job)     # todo: timeout
                time.sleep(rnd.uniform(1, 5))    # short sleep between enqueued jobs to make console more readable
                if self.state == JobRunner.State.stopping:
                    break
            q.join()
        finally:
            for _ in range(self.parallel_jobs):
                q.put(None)     # stopping workers
            for thread in threads:
                thread.join()
            if not self.done_async:
                for job in self.jobs:
                    job.done()

