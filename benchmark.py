#!/usr/bin/python3

from aws.AutoMLBenchmark import AutoMLBenchmark
from aws.AwsDockerOMLRun import AwsDockerOMLRun

import sys
import os
import json

framework = sys.argv[1]
benchmark = sys.argv[2]
mode = sys.argv[3]
apikey = os.popen("cat ~/.openml/config | grep apikey").read().split("=")[1][:-1]


with open("resources/benchmarks.json") as file:
  benchmarks = json.load(file)

with open("resources/frameworks.json") as file:
  frameworks = json.load(file)


bench = AutoMLBenchmark(benchmarks = benchmarks[benchmark], framework = frameworks[framework], openml_apikey = apikey)

print("Running `%s` on `%s` benchmarks in `%s` mode" % (framework, benchmark, mode))

if sys.argv[3] == "aws":
  bench.updateDockerContainer(upload = True)
  res = bench.runAWS(ssh_key = "laptop", sec_group = "launch-wizard-7", aws_instance_image = "ami-58d7e821")
else:
  bench.updateDockerContainer(upload = False)
  res = bench.runLocal()
