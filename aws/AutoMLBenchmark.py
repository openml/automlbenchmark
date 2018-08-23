#!/usr/bin/python3

import os
import re
import time
from aws.AwsDockerOMLRun import AwsDockerOMLRun

class AutoMLBenchmark:

  token = "THIS_IS_A_DUMMY_TOKEN"
  query_freq = 10

  def __init__(self, benchmarks, framework):
    self.benchmarks = benchmarks
    self.framework = framework

  def getContainerName(self):
    docker_image = self.framework["docker_image"]
    return "%s/%s:%s" % (docker_image["author"],  docker_image["image"], docker_image["tag"])

  def updateDockerContainer(self, upload = False):
    os.system("(cd docker && ./generate_docker.sh %s)" % (self.framework["dockerfile_folder"]))
    os.system("(cd docker && docker build -t %s -f %s/Dockerfile .)" % (self.getContainerName(), self.framework["dockerfile_folder"]))

    if upload:
      os.system("docker login")
      os.system("docker push %s" % (self.getContainerName()))

  def runLocal(self, keep_logs = False):
    results = []
    for benchmark in self.benchmarks:
      for fold in range(benchmark["folds"]):
        raw_log = os.popen("docker run --rm %s -f %i -t %i -s %i -p %i -m %s"  % (self.getContainerName(), fold, benchmark["openml_task_id"], benchmark["runtime"], benchmark["cores"], benchmark["metric"])).read()
        res = [x for x in raw_log.splitlines() if re.search(self.token, x)]
        if len(res) != 1:
            print("Fold %s on benchmark %s finished without valid result!" % (fold, benchmark["benchmark_id"]))
            res = 'nan'
        else:
            res = res[0].split(" ")[-1]
        results.append({"result":float(res), "benchmark_id":benchmark["benchmark_id"], "fold":fold})
        if keep_logs:
          results[-1]["log"] = raw_log

    return results

  def runAWS(self, ssh_key, sec_group, aws_instance_image, keep_logs = False):

      jobs = []
      for benchmark in self.benchmarks:
        for fold in range(benchmark["folds"]):
          jobs.append({
                      "benchmark_id":benchmark["benchmark_id"],
                      "fold":fold,
                      "run":AwsDockerOMLRun(
                                            ssh_key,
                                            sec_group,
                                            benchmark["aws_instance_type"],
                                            aws_instance_image,
                                            self.getContainerName(),
                                            benchmark["openml_task_id"],
                                            fold,
                                            benchmark["runtime"],
                                            benchmark["cores"],
                                            benchmark["metric"]
                                            )
                      })
      n_jobs = len(jobs)
      n_done = 0
      print("Created %s jobs\nStarting instances" % (n_jobs))
      for job in jobs:
        job["run"].createInstanceRun()
        job["result"] = job["run"].getResult()

      while n_done != n_jobs:
        time.sleep(self.query_freq)
        for job in jobs:
          job["result"] = job["run"].getResult()
        n_done = n_jobs - [job["result"] for job in jobs].count(None)
        print("%i/%i jobs done" % (n_done, n_jobs))

      if not keep_logs:
        for job in jobs:
          job["result"] = job["result"]["res"]

      print("All jobs done!\nTerminating Instances:")
      for job in jobs:
        job["run"].terminateInstance()
        del(job["run"])

      return jobs


if __name__ == "main":

  import json

  key = "laptop" #ssh key
  sec = "launch-wizard-7" # security group
  image = "ami-0615f1e34f8d36362" # aws instance image

  with open("resources/benchmarks.json") as file:
    benchmarks = json.load(file)

  with open("resources/frameworks.json") as file:
    frameworks = json.load(file)

  bench = AutoMLBenchmark(benchmarks = benchmarks["test_larger"], framework = frameworks["autosklearn"])
  bench.getContainerName()
  bench.updateDockerContainer(upload = True)
  res = bench.runLocal()
  res = bench.runLocal(keep_logs = True)
  bench.runAWS(ssh_key = key, sec_group = sec, aws_instance_image = image)
  res = bench.runAWS(ssh_key = key, sec_group = sec, aws_instance_image = image, keep_logs = True)
