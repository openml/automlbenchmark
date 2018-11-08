from enum import Enum
from importlib import import_module
import logging
import os

from .openml import Openml
from .resources import Resources
from .utils import available_memory_mb


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

    def __init__(self, framework_name, benchmark_name, config):
        """

        :param framework_name:
        :param benchmark_name:
        :param config:
        """
        self.resources = Resources(config)
        self.framework_name = framework_name
        self.framework_def = self.resources.framework_definition(framework_name)
        self.benchmark_name = benchmark_name
        self.benchmark_def = self.resources.benchmark_definition(benchmark_name)

        self.framework_module = import_module('automl.frameworks.'+self.framework_def.name)

    def setup(self, mode: SetupMode):
        """
        ensure all dependencies needed by framework are available
        and possibly download them if necessary.
        Delegates specific setup to the framework module
        """
        Benchmark.task_loader = Openml(api_key=self.resources.config.openml_apikey, cache_dir=self.resources.config.input_dir)
        if mode == Benchmark.SetupMode.skip or not hasattr(self.framework_module, 'setup'):
            return

        self.framework_module.setup()

    def run(self):
        """
        runs the framework for every task in the benchmark definition
        """
        for task_def in self.benchmark_def:
            self._run_task(task_def)

    def _run_task(self, task_def):
        """
        run the framework for every fold in the task definition
        :param task_def:
        """
        if Benchmark._is_task_disabled(task_def):
            return
        for fold in range(task_def.folds):
            self._run_fold(task_def, fold)

    def _run_fold(self, task_def, fold: int):
        """
        runs the framework against a given fold
        :param task_def: the task to run
        :param fold: the specific fold to use on this task
        """
        bench_task = BenchmarkTask(task_def, fold, self.resources.config)
        bench_task.load_data()
        bench_task.run(self.framework_module)

    def run_one(self, task_name: str, fold: int):
        """

        :param task_name:
        :param fold:
        """
        task_def = next(task for task in self.benchmark_def if task.name == task_name)
        if not task_def:
            raise ValueError("incorrect task name: {}".format(task_name))
        if fold >= task_def.folds:
            raise ValueError("fold value {} is out of range for task {}".format(fold, task_name))
        if Benchmark._is_task_disabled(task_def):
            raise ValueError("task {} is disabled, please enable it first".format(task_name))
        self._run_fold(task_def, fold)

    @property
    def _framework_dir(self):
        return os.path.dirname(self.framework_module.__file__)

    @staticmethod
    def _is_task_disabled(task_def):
        return hasattr(task_def, 'disabled') and task_def.disabled


class TaskConfig:

    def __init__(self, name, fold, metric, max_runtime_seconds,
                 cores, max_mem_size_mb,
                 input_dir, output_dir):
        self.name = name
        self.fold = fold
        self.metric = metric
        self.max_runtime_seconds = max_runtime_seconds
        self.cores = cores
        self.max_mem_size_mb = max_mem_size_mb
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_file_template = os.path.join(output_dir, "{{framework}}_{task}_{fold}".format(task=name, fold=fold))

    @staticmethod
    def from_def(task_def, fold, config):
        # todo: check available memory with possible warning
        return TaskConfig(
            name=task_def.name,
            fold=fold,
            metric=task_def.metric,
            max_runtime_seconds=task_def.runtime,
            cores=task_def.cores,
            max_mem_size_mb=config.max_mem_size_mb,
            input_dir=config.input_dir,
            output_dir=config.output_dir
        )


class BenchmarkTask:
    """

    """

    def __init__(self, task_def, fold, config):
        """

        :param task_def:
        :param fold:
        :param config:
        """
        self._task_def = task_def
        self._dataset = None
        self.fold = fold
        self.task = TaskConfig.from_def(self._task_def, self.fold, config)

    def load_data(self):
        """
        Loads the training dataset for the given task
        :param task: the task for which we want to load the dataset
        :return: path to the dataset file
        """
        if hasattr(self._task_def, 'openml_task_id'):
            self._dataset = Benchmark.task_loader.load(self._task_def.openml_task_id, self.fold)
        elif hasattr(self._task_def, 'dataset'):
            #todo
            raise NotImplementedError("raw dataset are not supported yet")
        else:
            raise ValueError("tasks should have one property among [openml_task_id, dataset]")

    def run(self, framework):
        """

        :param framework:
        :return:
        """
        self.task.output_file_template = self.task.output_file_template.format(framework=framework.__name__.rsplit('.', 1)[1].lower())
        framework.run(self._dataset, self.task)
        # todo: score predictions and print results
        # predictions_file = self.task.output_file_template + '.pred'
        # with open(predictions_file) as file:
        #     pass




