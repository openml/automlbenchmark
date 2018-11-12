import logging
from .benchmark import Benchmark


log = logging.getLogger(__name__)


class AWSBenchmark(Benchmark):
    """AWSBenchmark
    an extension of Benchmark class, to run benchmarks on AWS
    """

    def __init__(self, framework_name, benchmark_name, resources, region=None, reuse_instance=False):
        """

        :param framework_name:
        :param benchmark_name:
        :param resources:
        :param region:
        :param reuse_instance:
        """
        super().__init__(framework_name, benchmark_name, resources)
        self.region = region if region else self.resources.config['aws']['default_region']
        self.reuse_instance = reuse_instance

        ami = self.resources.config['aws']['regions'][self.region]['ami']
        if ami is None:
            raise ValueError("Region not supported by AMI yet.")

    def setup(self, mode):
        if mode == Benchmark.SetupMode.skip:
            return

        # todo: setup connection to EC2

    def run(self):
        if self.reuse_instance:
            self.start_instance("python {script} {framework} {benchmark} -m docker".format(
                script=self.resources.config['script'],
                framework=self.framework_def.name,
                benchmark=self.benchmark_name
            ))
        else:
            super().run()

    def _run_fold(self, task_def, fold: int):
        self.run_one(task_def.name, fold)

    def run_one(self, task_name: str, fold: int):
        self.start_instance("python {script} {framework} {benchmark} -m docker -t {task} -f {fold}".format(
            script=self.resources.config['script'],
            framework=self.framework_def.name,
            benchmark=self.benchmark_name,
            task=task_name,
            fold=fold
        ))

    def start_instance(self, cmd):
        pass


    def stop_instance(self, cmd):
        pass


