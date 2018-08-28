#!/usr/bin/python3

from aws.AutoMLBenchmark import AutoMLBenchmark

import argparse
import sys
import json
from time import time
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('framework', type=str,
                    help='The framework to evaluate as specified in /resources/frameworks.json')
parser.add_argument('benchmark', type=str,
                    help='The benchmark to run as specified in /resources/benchmarks.json')
parser.add_argument('mode', type=str,
                    help='The mode that specifies what backend is used (currently local or aws)')
parser.add_argument('-o', '--outfile', dest='outfile', metavar='-o', default=None,
                    help='Path where the output should be written.')
parser.add_argument('-r', '--region', dest='region', metavar='-r', default=None,
                    help='The region on which to run the benchmark when using AWS.')
args = parser.parse_args()


with open("resources/benchmarks.json") as file:
    benchmarks = json.load(file)

with open("resources/frameworks.json") as file:
    frameworks = json.load(file)


bench = AutoMLBenchmark(benchmarks=benchmarks[args.benchmark], framework=frameworks[args.framework], region_name=args.region)

print("Running `%s` on `%s` benchmarks in `%s` mode" % (args.framework, args.benchmark, args.mode))

if args.mode == "aws":
    bench.update_docker_container(upload=True)
    res = bench.run_aws()
elif args.mode == "local":
    bench.update_docker_container(upload=False)
    res = bench.run_local()
else:
    raise ValueError('mode must be one of \'aws\' or \'local\'.')

if args.outfile is not None:
    with open(args.outfile, "a") as file:
        for r in res:
            file.writelines(",".join([r["benchmark_id"],
                                     args.framework,
                                     frameworks[args.framework]["version"],
                                     str(r["fold"]),
                                     str(r["result"]),
                                     args.mode,
                                     str(int(time()))]) + "\n")
else:
    print(pd.DataFrame(res))
