import logging
import os
import re

from .utils import Namespace, json_load, lazy_property, memoize


log = logging.getLogger(__name__)


class Resources:

    @staticmethod
    def _normalize(config: Namespace):
        normalized = config.clone()
        for k, v in config:
            if re.search(r'_(dir|file)$', k):
                normalized[k] = os.path.realpath(os.path.expanduser(v))
        return normalized

    def __init__(self, config: Namespace):
        self.config = Resources._normalize(config)
        self.config.predictions_dir = os.path.join(self.config.output_dir, 'predictions')
        self.config.scores_dir = os.path.join(self.config.output_dir, 'scores')
        os.makedirs(self.config.predictions_dir, exist_ok=True)
        os.makedirs(self.config.scores_dir, exist_ok=True)
        log.debug("Normalized config: %s", self.config)

    def framework_definition(self, name):
        """

        :param name:
        :return: the framework definition with the given name
        """
        framework = self._frameworks[name]
        if not framework:
            raise ValueError("incorrect framework: {}".format(name))
        return framework

    @lazy_property
    def _frameworks(self):
        frameworks_file = self.config.frameworks_definition_file
        log.debug("loading frameworks definitions from %s", frameworks_file)
        with open(frameworks_file) as file:
            frameworks = json_load(file, as_object=True)
        for name, framework in frameworks:
            framework.name = name
            self._validate_framework(framework)
        return frameworks

    @memoize
    def benchmark_definition(self, name):
        """

        :param name:
        :return:
        """
        benchmark_file = "{dir}/{name}.json".format(dir=self.config.benchmarks_definition_dir, name=name)
        log.debug("loading benchmark definitions from %s", benchmark_file)
        if not os.path.exists(benchmark_file):
            benchmark_file = name
        if not os.path.exists(benchmark_file):
            raise ValueError("incorrect benchmark name or path: {}".format(name))

        with open(benchmark_file) as file:
            tasks = json_load(file, as_object=True)
        for task in tasks:
            self._validate_task(task)
        return tasks

    def _validate_framework(self, framework):
        # todo: validate docker image definition? anything else?
        pass

    def _validate_task(self, task):
        missing = []
        for config in ['name', 'openml_task_id', 'metric']:
            if task[config] is None:
                missing.append(config)
        if len(missing) > 0:
            raise ValueError("{missing} mandatory properties as missing in task definition {taskdef}".format(missing=missing, taskdef=task))

        for config in ['max_runtime_seconds', 'cores', 'folds']:
            if task[config] is None:
                task[config] = self.config.benchmark_definition_defaults[config]
                log.debug("config `{config}` not set for task {name}, using default `{value}`".format(config=config, name=task.name, value=task[config]))

        config = 'ec2_instance_type'
        if task[config] is None:
            task[config] = self.config.aws.ec2.instance_type
            log.debug("config `{config}` not set for task {name}, using default `{value}`".format(config=config, name=task.name, value=task[config]))


__INSTANCE__: Resources = None


def from_config(config: Namespace):
    global __INSTANCE__
    __INSTANCE__ = Resources(config)


def get() -> Resources:
    return __INSTANCE__


def config():
    return __INSTANCE__.config

