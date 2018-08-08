#!/usr/bin/python3

import os
import re
import time
from aws.AwsDockerOMLRun import AwsDockerOMLRun

class AutoMLBenchmark:

  token = "THIS_IS_A_DUMMY_TOKEN"

  def __init__(self, benchmarks, framework, openml_apikey):
    self.benchmarks = benchmarks
    self.framework = framework
    self.openml_apikey = openml_apikey

  def getContainerName(self):
    docker_image = self.framework["docker_image"]
    return "%s/%s:%s" % (docker_image["author"],  docker_image["image"], docker_image["tag"])

  def updateDockerContainer(self, upload = False):
    #os.system("(cd docker && ./generate_docker.sh %s)" % (self.framework["dockerfile_folder"]))
    os.system("docker build -t %s docker/%s" % (self.getContainerName(), self.framework["dockerfile_folder"]))

    if upload:
      os.system("docker login")
      os.system("docker push %s" % (self.getContainerName()))

  def runLocal(self):
    results = {}
    for benchmark in self.benchmarks:
      res = os.popen("docker run --rm %s %s %s %s %s" % (self.getContainerName(), benchmark["id"], benchmark["runtime"], benchmark["cores"], self.openml_apikey)).read()
      res = [x for x in res.splitlines() if re.search(self.token, x)]
      if len(res) != 1:
          print("Run on task %s finished without valid result!" %s (benchmark["id"]))
          res = float('nan')
      else:
          res = res[0].split(" ")[-1]
      results[benchmark["id"]] = res

    return results

  def runAWS(self, ssh_key, sec_group, aws_instance_image):

      runs = [AwsDockerOMLRun(ssh_key, sec_group, b["aws_instance_type"], aws_instance_image, self.getContainerName(), b["id"], b["runtime"], b["cores"], self.openml_apikey) for b in self.benchmarks]

      for run in runs:
        run.createInstanceRun()

      results = [run.getResult() for run in runs]

      while None in results:
        time.sleep(10)
        results = [run.getResult() for run in runs]
        print("%i/%i benchmarks done" % (len(results) - results.count(None), len(results)))

      print("all benchmarks done!")
      [run.terminateInstance() for run in runs]
      return dict(zip([x["id"] for x in self.benchmarks], results))


if __name__ == "main":

  import json
  from AwsDockerOMLRun import AwsDockerOMLRun

  key = "laptop" #ssh key
  sec = "launch-wizard-7" # security group
  image = "ami-58d7e821" # aws instance image
  apikey = os.popen("cat ~/.openml/config | grep apikey").read().split("=")[1][:-1] # openml apikey

  with open("../resources/benchmarks.json") as file:
    benchmarks = json.load(file)

  with open("../resources/frameworks.json") as file:
    frameworks = json.load(file)

  bench = AutoMLBenchmark(benchmarks = benchmarks["test"], framework = frameworks["randomForest"], openml_apikey = apikey)
  bench.getContainerName()
  bench.updateDockerContainer(upload = False)
  bench.runLocal()
  bench.runAWS(ssh_key = key, sec_group = sec, aws_instance_image = image)
