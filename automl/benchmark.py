from importlib import import_module

from .openml import OpenML


task_loader = OpenML()

class Benchmark(object):
    """
    Structure containing the generic information needed to run a benchmark:
     - the datasets
     - the automl framework


     we need to support:
     - openml datasets
     - openml tasks
     - openml studies (=benchmark suites)
     - user-defined (list of) datasets
    """

    def __init__(self, framework, tasks):
        super().__init__()
        self.framework_desc = framework
        self.framework_module = import_module('automl.frameworks.'+self.framework.name)
        self.tasks_desc = tasks

    def setup(self):
        """
        ensure all dependencies needed by framework are available
        and possibly download them if necessary.
        Delegates specific setup to the framework module
        """
        self.framework_module.setup()

    def run(self):
        """
        runs the framework for every fold of every task
        """
        for task_desc in self.tasks_desc:
            for fold in range(task_desc.folds):
                self.run_one(task_desc, fold)

    def run_one(self, task_desc, fold):
        """
        runs the framework against a given fold
        """
        bench_task = BenchmarkTask(task_desc)
        bench_task.load_data(fold)
        bench_task.run(self.framework_module)


class Dataset(object):

    def __init__(self):
        super().__init__()
        self.train = None
        self.test = None
        self.x = None
        self.y = None


class BenchmarkTask(object):

    def __init__(self, task):
        super().__init__()
        self.task = task
        self.dataset = None

    def load_data(self, fold):
        """
        Loads the training dataset for the given task
        :param task: the task for which we want to load the dataset
        :return: path to the dataset file
        """
        if hasattr(self.task, 'openml_task_id'):
            self.dataset = task_loader.load(self.task.openml_task_id, fold)
        elif hasattr(self.task, 'dataset'):
            #todo
            pass
        else:
            raise ValueError("tasks should have one property among [openml_task_id, dataset]")

    def run(self, framework):
        framework.run(self.dataset, self.task)




