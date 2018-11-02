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
parser.add_argument('-o', '--outfolder', dest='outfolder', metavar='-o', default=None,
                    help='Path where the output should be written.')
parser.add_argument('-r', '--region', dest='region', metavar='-r', default=None,
                    help='The region on which to run the benchmark when using AWS.')
args = parser.parse_args()


with open("resources/benchmarks.json") as file:
    benchmarks = json.load(file)

with open("resources/frameworks.json") as file:
    frameworks = json.load(file)

with open("resources/ami.json") as file:
    ami = json.load(file)

if args.region is not None and args.region not in ami.keys():
    raise ValueError("Region not supported by AMI yet.")


bench = AutoMLBenchmark(benchmarks=benchmarks[args.benchmark], framework=frameworks[args.framework], region_name=args.region)

print("Running `%s` on `%s` benchmarks in `%s` mode" % (args.framework, args.benchmark, args.mode))

if args.mode not in ['aws', 'local']:
    raise ValueError("mode must be one of 'aws' or 'local'.")

bench.update_docker_container(upload=(args.mode == 'aws'))
res = bench.run(where=args.mode, log_directory=args.outfolder)
print(pd.DataFrame(res))
