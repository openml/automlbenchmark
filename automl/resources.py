import logging
import os

from .utils import dict_to_namedtuple, extend_namedtuple, json_load


log = logging.getLogger(__name__)


class Resources:

    @staticmethod
    def _normalize(config):
        normalized = config.copy()
        for k in ['frameworks_definition_file', 'benchmarks_definition_dir', 'input_dir', 'output_dir']:
            normalized[k] = os.path.expanduser(config[k])
        return normalized

    def __init__(self, config):
        self.config = Resources._normalize(config)

    def framework_definition(self, name):
        """

        :param name:
        :return: the framework definition with the given name
        """
        frameworks_file = self.config['frameworks_definition_file']
        log.debug("loading frameworks definitions from %s", frameworks_file)
        with open(frameworks_file) as file:
            frameworks = json_load(file)

        if not frameworks[name]:
            raise ValueError("incorrect framework: {}".format(name))

        framework = frameworks[name]
        framework['name'] = name
        framework = dict_to_namedtuple(framework, 'Framework')
        return framework

    def benchmark_definition(self, name):
        """

        :param name:
        :return:
        """
        benchmark_file = "{dir}/{name}.json".format(dir=self.config['benchmarks_definition_dir'], name=name)
        log.debug("loading benchmark definitions from %s", benchmark_file)
        if not os.path.exists(benchmark_file):
            benchmark_file = name
        if not os.path.exists(benchmark_file):
            raise ValueError("incorrect benchmark name or path: {}".format(name))

        with open(benchmark_file) as file:
            tasks = json_load(file, as_object=True)
        return tasks
