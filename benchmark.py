#!/usr/bin/python3

from aws.AutoMLBenchmark import AutoMLBenchmark

import sys
import json
from time import time
import pandas as pd

aws_instance_image = "ami-0615f1e34f8d36362"

framework = sys.argv[1]
benchmark = sys.argv[2]
mode = sys.argv[3]
outfile = sys.argv[4] if len(sys.argv) > 4 else None


with open("resources/benchmarks.json") as file:
    benchmarks = json.load(file)

with open("resources/frameworks.json") as file:
    frameworks = json.load(file)


bench = AutoMLBenchmark(benchmarks=benchmarks[benchmark], framework=frameworks[framework])#, region_name=outfile)

print("Running `%s` on `%s` benchmarks in `%s` mode" % (framework, benchmark, mode))

if mode == "aws":
    bench.update_docker_container(upload=True)
    res = bench.run_aws(aws_instance_image)
elif mode == "local":
    bench.update_docker_container(upload=False)
    res = bench.run_local()

if outfile is not None:
    with open(outfile, "a") as file:
        for r in res:
            file.writelines(",".join([r["benchmark_id"],
                                     framework,
                                     frameworks[framework]["version"],
                                     str(r["fold"]),
                                     str(r["result"]),
                                     mode,
                                     str(int(time()))]) + "\n")
else:
    print(pd.DataFrame(res))
