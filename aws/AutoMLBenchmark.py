#!/bin/python3
from AwsDockerOMLRun import AwsDockerOMLRun

import os
import re
import time

class AutoMLBenchmark:

  token = "THIS_IS_A_DUMMY_TOKEN"

  def __init__(self, benchmarks, framework, ssh_key, sec_group, aws_instance_image, openml_apikey):
    self.benchmarks = benchmarks
    self.framework = framework
    self.ssh_key = ssh_key
    self.sec_group = sec_group
    self.aws_instance_image = aws_instance_image
    self.openml_apikey = openml_apikey

  def getContainerName(self):
    docker_image = self.framework["docker_image"]
    return "%s/%s:%s" % (docker_image["author"],  docker_image["image"], docker_image["tag"])

  def updateDockerContainer(self, upload = False):
    os.system("docker build -t %s %s" % (self.getContainerName(), self.framework["dockerfile"]))

    if upload:
      os.system("docker login")
      os.system("docker upload %s" % (self.getContainerName()))

  def runLocal(self):
    results = {}
    for benchmark in self.benchmarks:
      res = os.popen("docker run --rm %s %s %s %s %s" % (self.getContainerName(), benchmark["id"], benchmark["runtime"], benchmark["cores"], self.openml_apikey)).read()
      results[benchmark["id"]] = [x for x in res.splitlines() if re.search(self.token, x)][0].split(" ")[-1]
    return results

  def runAWS(self):

      runs = [AwsDockerOMLRun(self.ssh_key, self.sec_group, b["aws_instance_type"], self.aws_instance_image, self.getContainerName(), b["id"], b["runtime"], b["cores"], self.openml_apikey) for b in self.benchmarks]

      for run in runs:
        run.createInstanceRun()

      results = [run.getResult() for run in runs]

      while None in results:
        time.sleep(10)
        results = [run.getResult() for run in runs]
        print("%i/%i benchmarks done" % (len(results) - results.count(None), len(results)))

      print("done!")
      [run.terminateInstance() for run in runs]
      return dict(zip([x["id"] for x in self.benchmarks], results))


if __name__ == "main":

  import json

  key = "laptop" #ssh key
  sec = "launch-wizard-7" # security group
  image = "ami-58d7e821" # aws instance image
  apikey = os.popen("cat ~/.openml/config | grep apikey").read().split("=")[1][:-1] # openml apikey

  with open("../resources/benchmarks.json") as file:
    benchmarks = json.load(file)

  with open("../resources/frameworks.json") as file:
    frameworks = json.load(file)

  bench = AutoMLBenchmark(benchmarks = benchmarks["test"], framework = frameworks["randomForest"], ssh_key = key, sec_group = sec, aws_instance_image = image, openml_apikey = apikey)
  bench.getContainerName()
  bench.framework["dockerfile"] = "../docker/RandomForest"
  bench.updateDockerContainer()
  bench.runLocal()
  bench.runAWS()
