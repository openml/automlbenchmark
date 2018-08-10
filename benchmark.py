#!/usr/bin/python3

from aws.AutoMLBenchmark import AutoMLBenchmark
from aws.AwsDockerOMLRun import AwsDockerOMLRun

import sys
import os
import json
from time import time
import pandas as pd


# changeme
ssh_key = "laptop"
sec_group = "launch-wizard-7"
aws_instance_image = "ami-58d7e821"


framework = sys.argv[1]
benchmark = sys.argv[2]
mode = sys.argv[3]
outfile = sys.argv[4] if len(sys.argv) > 4 else None


with open("resources/benchmarks.json") as file:
  benchmarks = json.load(file)

with open("resources/frameworks.json") as file:
  frameworks = json.load(file)


bench = AutoMLBenchmark(benchmarks = benchmarks[benchmark], framework = frameworks[framework], openml_apikey = "c1994bdb7ecb3c6f3c8f3b35f4b47f1f")

print("Running `%s` on `%s` benchmarks in `%s` mode" % (framework, benchmark, mode))

if mode == "aws":
  bench.updateDockerContainer(upload = True)
  res = bench.runAWS(ssh_key, sec_group, aws_instance_image)
elif mode == "local":
  bench.updateDockerContainer(upload = False)
  res = bench.runLocal()

if outfile is not None:
    with open(outfile, "a") as file:
        for r in res:
            file.writelines(",".join([r["benchmark_id"],
                                     str(r["fold"]),
                                     str(r["result"]),
                                     mode,
                                     str(int(time()))]) + "\n")
else:
    print(pd.DataFrame(res))
