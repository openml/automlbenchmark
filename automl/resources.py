import logging
import os
import re

from .utils import Namespace, json_load


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
        frameworks_file = self.config.frameworks_definition_file
        log.debug("loading frameworks definitions from %s", frameworks_file)
        with open(frameworks_file) as file:
            frameworks = json_load(file, as_object=True)

        if not frameworks[name]:
            raise ValueError("incorrect framework: {}".format(name))

        framework = frameworks[name]
        framework.name = name
        return framework

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
        return tasks
