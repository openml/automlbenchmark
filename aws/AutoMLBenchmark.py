#!/usr/bin/python3

import os
import re
import time
from aws.AwsDockerOMLRun import AwsDockerOMLRun


class AutoMLBenchmark:
    token = "6744dfceeb4d2b4a9e60874bcd46b3a1"
    overhead_time = 10 * 60 #additional time for setup etc.

    def __init__(self, benchmarks, framework, query_frequency=10, region_name=None):
        self.benchmarks = benchmarks
        self.framework = framework
        self.query_frequency = query_frequency
        self.region_name = region_name


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

    def run_aws(self, aws_instance_image, keep_logs=False):

        jobs = []
        for benchmark in self.benchmarks:
            for fold in range(benchmark["folds"]):
                jobs.append({
                    "benchmark_id": benchmark["benchmark_id"],
                    "fold": fold,
                    "run": AwsDockerOMLRun(
                        benchmark["aws_instance_type"],
                        aws_instance_image,
                        self.get_container_name(),
                        benchmark["openml_task_id"],
                        fold,
                        benchmark["runtime"],
                        benchmark["cores"],
                        benchmark["metric"],
                        self.region_name
                    )
                })
        n_jobs = len(jobs)
        n_done = 0
        print("Created %s jobs\nStarting instances" % (n_jobs))
        for job in jobs:
            job["run"].createInstanceRun()
            job["result"] = job["run"].getResult()
        start_time = time.time()
        while n_done != n_jobs:
            time.sleep(self.query_frequency)
            runtime = int(time.time() - start_time)
            minutes, seconds = divmod(runtime, 60)
            hours, minutes = divmod(minutes, 60)
            for job in jobs:
                job["result"] = job["run"].getResult()
                if job["result"] is None and runtime > (job["run"].runtime + self.overhead_time):
                    print("Benchmark %s on fold %i hit the walltime and is terminated" % (job["benchmark_id"], job["fold"]))
                    job["run"].terminateInstance()
                    job["result"] = {"log":"hit walltime", "res":"nan"}
            n_done = n_jobs - [job["result"] for job in jobs].count(None)
            print("[%02d:%02d:%02d] - %i/%i jobs done" % (hours, minutes, seconds, n_done, n_jobs))


        if not keep_logs:
            for job in jobs:
                job["result"] = job["result"]["res"]

        print("All jobs done!\nTerminating Instances:")
        for job in jobs:
            job["run"].terminateInstance()
            del (job["run"])

        return jobs


if __name__ == "main":
    import json

    with open("resources/benchmarks.json") as file:
        benchmarks = json.load(file)

    with open("resources/frameworks.json") as file:
        frameworks = json.load(file)

    bench = AutoMLBenchmark(benchmarks=benchmarks["test_larger"], framework=frameworks["TPOT"])
    bench.get_container_name()
    bench.update_docker_container(upload=True)
    res = bench.run_local()
    res = bench.run_local(keep_logs=True)
    bench.run_aws(aws_instance_image=image)
    res = bench.run_aws(aws_instance_image=image, keep_logs=True)
