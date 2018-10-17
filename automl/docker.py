from automl.benchmark import Benchmark


class DockerBenchmark(Benchmark):

    def __init__(self, framework, tasks, keep_instance=False):
        super().__init__(framework, tasks)
        self.keep_instance = keep_instance

    def setup(self):
        """
        build the docker image for given framework
        """
        super().setup()

    def run(self):
        if self.keep_instance:
            self.start_docker("python {script} {framework} {benchmark} local".format(...))
        else:
            super().run()

    def run_one(self, task_desc, fold):
        self.start_docker("python {script} {framework} {benchmark} local -t {task} -f {fold}".format(...))

    def start_docker(self, cmd):
        pass