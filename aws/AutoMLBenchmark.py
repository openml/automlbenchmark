#!/usr/bin/python3

import os
import re
import time
import json
from aws.AwsDockerOMLRun import AwsDockerOMLRun


class AutoMLBenchmark:

    def __init__(self, benchmarks, framework, region_name=None):
        self.benchmarks = benchmarks
        self.framework = framework
        self.region_name = region_name

        # load config file
        with open("config.json") as file:
            config = json.load(file)

        self.token = config["token"]
        self.overhead_time = config["overhead_time"]
        self.query_frequency = config["query_frequency"]
        self.max_parallel_jobs = config["max_parallel_jobs"]

    def get_container_name(self):
        docker_image = self.framework["docker_image"]
        return "%s/%s:%s" % (docker_image["author"], docker_image["image"], docker_image["tag"])

    def update_docker_container(self, upload=False):
        os.system("(cd docker && ./generate_docker.sh %s)" % (self.framework["dockerfile_folder"]))
        os.system("(cd docker && docker build -t %s -f %s/Dockerfile .)" % (
            self.get_container_name(), self.framework["dockerfile_folder"]))

        if upload:
            os.system("docker login")
            os.system("docker push %s" % (self.get_container_name()))

    def run(self, where='local', log_directory=None):
        """ Runs the benchmark either locally or on AWS, and optionally stores the log.

        :param where: string (default: 'local')
         Specifies where to run the benchmark, either 'local' or 'aws'.
        :param log_directory: string or None (default: None)
         Path to the directory where logs should be saved. If None, logs are not stored.
         If a directory for the path is not found, it will be created.
        :return: results
        """
        if log_directory is not None:
            if not os.path.exists(log_directory):
                os.mkdir(log_directory)

        if where == 'local':
            return self.run_local(log_directory)
        elif where == 'aws':
            return self.run_aws(log_directory)
        else:
            raise ValueError("`where` can only be one of 'local' or 'aws'.")

    def _secure_log_filepath_for_benchmark(self, log_directory, task_id, fold):
        """ Compose log filepath for given directory, task, fold. Rename any existing file with that name.

        :param log_directory: string. directory the log file is to be written in.
        :param task_id: string or int. OpenML task id.
        :param fold: string or int. fold number for the task.
        :return: string. the file path for the log file.
        """
        log_path = os.path.join(log_directory,
                                'log_{}_{}_{}.txt'.format(self.framework['dockerfile_folder'], task_id, fold))
        if os.path.exists(log_path):
            print("! WARNING ! Old log files exist. Renaming them with a '.old' extension."
                  "Should there be any old '.old'-logs, they will be deleted.")
            os.rename(log_path, log_path[:-4] + '.old')
        return log_path

    def run_local(self, log_directory=None):
        results = []
        for benchmark in self.benchmarks:
            for fold in range(benchmark["folds"]):
                raw_log = os.popen("docker run --rm %s -f %i -t %i -s %i -p %i -m %s" % (
                    self.get_container_name(), fold, benchmark["openml_task_id"], benchmark["runtime"], benchmark["cores"],
                    benchmark["metric"])).read()
                res = [x for x in raw_log.splitlines() if re.search(self.token, x)]
                if len(res) != 1:
                    print("Fold %s on benchmark %s finished without valid result!" % (fold, benchmark["benchmark_id"]))
                    res = 'nan'
                else:
                    res = res[0].split(" ")[-1]
                results.append({"result": float(res), "benchmark_id": benchmark["benchmark_id"], "fold": fold})

                if log_directory is not None:
                    log_path = self._secure_log_filepath_for_benchmark(log_directory, benchmark["openml_task_id"], fold)
                    with open(log_path, 'w') as fh:
                        fh.write(raw_log)

        return results

    def run_aws(self, log_directory=None):

        jobs = []

        for benchmark in self.benchmarks:
            for fold in range(benchmark["folds"]):
                if log_directory is not None:
                    log_filepath = self._secure_log_filepath_for_benchmark(log_directory,
                                                                           benchmark["openml_task_id"],
                                                                           fold)
                else:
                    log_filepath = None
                jobs.append({"benchmark_id": benchmark["benchmark_id"],
                            "fold": fold, "run": AwsDockerOMLRun(
                                                                 benchmark["aws_instance_type"],
                                                                 self.get_container_name(),
                                                                 benchmark["openml_task_id"],
                                                                 fold,
                                                                 benchmark["runtime"],
                                                                 benchmark["cores"],
                                                                 benchmark["metric"],
                                                                 self.region_name,
                                                                 log_filepath=log_filepath)})

        def chunk_jobs(l, n):
            for i in range(0, len(l), n):
                yield l[i:i + n]

        chunks = list(chunk_jobs(jobs, self.max_parallel_jobs))
        print("Grouping %i jobs in %i chunk(s) of %i parallel jobs" % (len(jobs),
                                                                       len(chunks),
                                                                       min(len(jobs), self.max_parallel_jobs)))

        for ind, chunk in enumerate(chunks):
            print("---- Chunk %i/%i ----" % (ind + 1, len(chunks)))
            n_jobs = len(chunk)
            n_done = 0
            if ind > 0:
                time.sleep(30)  # wait for previous instances to be shut down
            print("Created %s jobs\nStarting instances" % (n_jobs))
            for job in chunk:
                job["run"].createInstanceRun()
                job["result"] = job["run"].getResult()
            start_time = time.time()
            while n_done != n_jobs:
                time.sleep(self.query_frequency)
                runtime = int(time.time() - start_time)
                minutes, seconds = divmod(runtime, 60)
                hours, minutes = divmod(minutes, 60)
                for job in chunk:
                    job["result"] = job["run"].getResult()
                    if job["result"] is None and runtime > (job["run"].runtime + self.overhead_time):
                        print("Benchmark %s on fold %i hit the walltime and is terminated" % (job["benchmark_id"], job["fold"]))
                        job["run"].terminateInstance()
                        job["result"] = {"log": "hit walltime", "res": "nan"}
                n_done = n_jobs - [job["result"] for job in chunk].count(None)
                print("[%02d:%02d:%02d] - %i/%i jobs done" % (hours, minutes, seconds, n_done, n_jobs))

            print("Chunk %i done, terminating Instances:" % (ind + 1))
            for job in chunk:
                job["run"].terminateInstance()
                del (job["run"])

        jobs = [job for chunk in chunks for job in chunk]
        job["result"] = job["result"]["res"]
        return jobs


if __name__ == "main":

    with open("resources/benchmarks.json") as file:
        benchmarks = json.load(file)

    with open("resources/frameworks.json") as file:
        frameworks = json.load(file)

    bench = AutoMLBenchmark(benchmarks=benchmarks["test_larger"], framework=frameworks["RandomForest"])
    bench.get_container_name()
    bench.update_docker_container(upload=True)
    res = bench.run_local()
    res = bench.run_local(keep_logs=True)
    bench.run_aws()
    res = bench.run_aws(keep_logs=True)
