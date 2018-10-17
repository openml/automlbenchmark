from automl.benchmark import Benchmark


class AWSBenchmark(Benchmark):

    def __init__(self, framework, tasks, region, keep_instance=False):
        super().__init__(framework, tasks)
        self.region = region
        self.keep_instance = keep_instance

    def setup(self):
        """
        setup connection to EC2

        """
        super().setup()

    def run(self):
        if self.keep_instance:
            self.start_instance("python {script} {framework} {benchmark} docker".format(...))
        else:
            super().run()

    def run_one(self, task_desc, fold):
        self.start_instance("python {script} {framework} {benchmark} docker -t {task} -f {fold}".format(...))

    def start_instance(self, cmd):
        pass


