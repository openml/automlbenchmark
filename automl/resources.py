import os

from .utils import dict_to_namedtuple, extend_namedtuple, json_load


class Resources:

    def __init__(self, config):
        self.config = config

    def framework_definition(self, name):
        """

        :param name:
        :return: the framework definition with the given name
        """
        with open(self.config['frameworks_definition_file']) as file:
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
        benchmark_file = "{dir}/{name}.json".format(dir=self.config['benchmarks_definitions_dir'], name=name)
        if not os.path.exists(benchmark_file):
            benchmark_file = name
        if not os.path.exists(benchmark_file):
            raise ValueError("incorrect benchmark name or path: {}".format(name))

        with open(benchmark_file) as file:
            tasks = json_load(file, as_object=True)
        return tasks
