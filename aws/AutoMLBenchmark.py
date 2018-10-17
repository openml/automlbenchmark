#!/usr/bin/python3

import os
import re
import sys
import time
import json
from aws.AwsDockerOMLRun import AwsDockerOMLRun

# sys.path.insert(0, os.getcwd())

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


    # def run_test(self, keep_logs=False):
    #     results = []
    #     for benchmark in self.benchmarks:
    #         for fold in range(benchmark["folds"]):
    #             task_id = benchmark['openml_task_id']
    #             load_log = os.popen("python -Wignore ./docker/common/load_data.py -t {task_id} -f {fold_n} --train ./docker/common/train.arff --test ./docker/common/test.arff"
    #                                 .format(task_id=task_id, fold_n=fold)
    #                                 ).read()
    #             run_log = os.popen("python -Wignore ./docker/{framework}/run.py {time} {threads} {metric} {memory}"
    #                                .format(
    #                                     framework=self.framework["dockerfile_folder"],
    #                                     time=benchmark["runtime"],
    #                                     threads=benchmark["cores"],
    #                                     metric=benchmark["metric"],
    #                                     memory="2048"
    #                                 )).read()
    #             eval_log = os.popen("python -Wignore ./docker/common/evaluate.py ./docker/common/test.arff ./docker/common/predictions.csv {metric}"
    #                                 .format(metric=benchmark['metric'])
    #                                 ).read()
    #             raw_log = '\n'.join([load_log, run_log, eval_log])
    #             res = [x for x in raw_log.splitlines() if re.search(self.token, x)]
    #             if len(res) != 1:
    #                 print("Fold %s on benchmark %s finished without valid result!" % (fold, benchmark["benchmark_id"]))
    #                 res = 'nan'
    #             else:
    #                 res = res[0].split(" ")[-1]
    #             results.append({"result": float(res), "benchmark_id": benchmark["benchmark_id"], "fold": fold})
    #             if keep_logs:
    #                 results[-1]["log"] = raw_log
    #
    #     return results


    def run_local(self, keep_logs=False):
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
                if keep_logs:
                    results[-1]["log"] = raw_log

        return results

    def run_aws(self, keep_logs=False):

        jobs = []
        for benchmark in self.benchmarks:
            for fold in range(benchmark["folds"]):
                jobs.append({"benchmark_id": benchmark["benchmark_id"],
                            "fold": fold, "run": AwsDockerOMLRun(
                                                                 benchmark["aws_instance_type"],
                                                                 self.get_container_name(),
                                                                 benchmark["openml_task_id"],
                                                                 fold,
                                                                 benchmark["runtime"],
                                                                 benchmark["cores"],
                                                                 benchmark["metric"],
                                                                 self.region_name)})

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

        if not keep_logs:
            for job in jobs:
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
