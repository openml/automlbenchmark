from copy import copy
from enum import Enum
from importlib import import_module
import logging
import os

from .openml import Openml
from .resources import get as rget, config as rconfig
from .results import Scoreboard, TaskResult
from .utils import available_memory_mb, now_iso, str2bool


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

    def __init__(self, framework_name: str, benchmark_name: str):
        """

        :param framework_name:
        :param benchmark_name:
        :param resources:
        """
        self.framework_name = framework_name
        self.framework_def = rget().framework_definition(framework_name)
        log.debug("Using framework definition: %s", self.framework_def)
        self.benchmark_name = benchmark_name
        self.benchmark_def = rget().benchmark_definition(benchmark_name)
        log.debug("Using benchmark definition: %s", self.benchmark_def)
        self.uid = "{}_{}_{}".format(framework_name, benchmark_name, now_iso(micros=True, no_sep=True))

        self.framework_module = import_module('automl.frameworks.'+self.framework_def.name)

    def setup(self, mode: SetupMode):
        """
        ensure all dependencies needed by framework are available
        and possibly download them if necessary.
        Delegates specific setup to the framework module
        """
        Benchmark.task_loader = Openml(api_key=rconfig().openml_apikey, cache_dir=rconfig().input_dir)
        if mode == Benchmark.SetupMode.skip or not hasattr(self.framework_module, 'setup'):
            return

        self.framework_module.setup()

    def cleanup(self):
        # anything to do?
        pass

    def run(self, save_scores=False):
        """
        runs the framework for every task in the benchmark definition
        """
        scores = []
        for task_def in self.benchmark_def:
            if Benchmark._is_task_enabled(task_def):
                scores.extend(self._run_task(task_def))

        if len(scores) == 0 or not any(scores):
            return None

        board = Scoreboard(scores, framework_name=self.framework_name, benchmark_name=self.benchmark_name)
        if save_scores:
            board.save(append=True)
        log.info("Summing up scores for {benchmark} benchmark:\n {scores}".format(
            benchmark=self.benchmark_name,
            scores=board.as_data_frame()
        ))
        return board.as_data_frame()

    def run_one(self, task_name: str, fold, save_scores=False):
        """

        :param task_name:
        :param fold:
        :param save_scores:
        """
        scores = []
        task_def = self._get_task_def(task_name)
        if fold is None:
            scores.extend(self._run_task(task_def))
        elif isinstance(fold, int):
            scores.append(self._run_fold(task_def, fold))
        elif isinstance(fold, list) and all(isinstance(f, int) for f in fold):
            for f in fold:
                scores.append(self._run_fold(task_def, f))
        else:
            raise ValueError("fold value should be None, an int, or a list of ints")

        if len(scores) == 0 or not any(scores):
            return None

        board = Scoreboard(scores, framework_name=self.framework_name, task_name=task_name)
        if save_scores:
            board.save(append=True)
        log.info("Summing up scores for {task} task:\n {scores}".format(
            task=task_name,
            scores=board.as_data_frame()
        ))
        return board.as_data_frame()

    def _run_task(self, task_def):
        """
        run the framework for every fold in the task definition
        :param task_def:
        """
        scores = []
        for fold in range(task_def.folds):
            scores.append(self._run_fold(task_def, fold))
        return scores

    def _run_fold(self, task_def, fold: int):
        """
        runs the framework against a given fold
        :param task_def: the task to run
        :param fold: the specific fold to use on this task
        """
        if fold < 0 or fold >= task_def.folds:
            raise ValueError("fold value {} is out of range for task {}".format(fold, task_def.name))

        bench_task = BenchmarkTask(task_def, fold)
        bench_task.load_data()
        return bench_task.run(self.framework_module)

    def _get_task_def(self, task_name):
        task_def = next(task for task in self.benchmark_def if task.name == task_name)
        if not task_def:
            raise ValueError("incorrect task name: {}".format(task_name))
        if not Benchmark._is_task_enabled(task_def):
            raise ValueError("task {} is disabled, please enable it first".format(task_name))
        return task_def

    @property
    def _framework_dir(self):
        return os.path.dirname(self.framework_module.__file__)

    @staticmethod
    def _is_task_enabled(task_def):
        return not hasattr(task_def, 'enabled') or str2bool(str(task_def.enabled))


class TaskConfig:

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

    @staticmethod
    def from_def(task_def, fold, config):
        # todo: check available memory with possible warning: cf. utils.available_memory_mb()
        return TaskConfig(
            name=task_def.name,
            fold=fold,
            metrics=task_def.metric,
            max_runtime_seconds=task_def.max_runtime_seconds,
            cores=task_def.cores,
            max_mem_size_mb=config.max_mem_size_mb,
            input_dir=config.input_dir,
            output_dir=config.predictions_dir,
        )


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

    def load_data(self):
        """
        Loads the training dataset for the current given task
        :return: path to the dataset file
        """
        if hasattr(self._task_def, 'openml_task_id'):
            self._dataset = Benchmark.task_loader.load(self._task_def.openml_task_id, self.fold)
            log.debug("loaded OpenML dataset for task_id %s", self._task_def.openml_task_id)
        elif hasattr(self._task_def, 'dataset'):
            # todo
            raise NotImplementedError("raw dataset are not supported yet")
        else:
            raise ValueError("tasks should have one property among [openml_task_id, dataset]")

    def run(self, framework):
        """

        :param framework:
        :return:
        """
        framework_name = framework.__name__.rsplit('.', 1)[1]
        results = TaskResult(task_name=self.task.name, fold=self.fold)
        task_config = copy(self.task)
        task_config.framework = framework_name
        task_config.output_predictions_file = results._predictions_file(task_config.framework.lower())
        try:
            framework.run(self._dataset, task_config)
        except Exception as e:
            log.error("%s failed with error $s", framework_name, str(e))
            log.exception(e)

        return results.compute_scores(framework_name, task_config.metrics)
