
import argparse
import json
from collections import namedtuple

import automl


def json_load_as_obj(file):
    return json.load(file, object_hook=lambda dic: namedtuple('JsonNode', dic.keys())(*dic.values()))


parser = argparse.ArgumentParser()
parser.add_argument('framework', type=str,
                    help='The framework to evaluate as specified in /resources/frameworks.json')
parser.add_argument('benchmark', type=str,
                    help='The benchmark to run as specified in /resources/benchmarks.json')
parser.add_argument('mode', type=str,
                    help='The mode that specifies what backend is used (currently local, docker, or aws)')
parser.add_argument('-t', '--task', dest='task', metavar='-t', default=None,
                    help='The specific task to run in the benchmark')
parser.add_argument('-f', '--fold', dest='fold', metavar='-f', type=int, default=None,
                    help='The specific fold to run in the benchmark')
parser.add_argument('-o', '--outfile', dest='outfile', metavar='-o', default=None,
                    help='Path where the output should be written.')
parser.add_argument('-r', '--region', dest='region', metavar='-r', default=None,
                    help='The region on which to run the benchmark when using AWS.')
parser.add_argument('-k', '--keep_instance', dest='keep_instance', metavar='-k', type=bool, default=False,
                    help='Set to true if reusing the same instance(s) for all tasks (docker and aws mode only)')
args = parser.parse_args()


with open("resources/benchmarks.json") as file:
    benchmarks = json_load_as_obj(file)

with open("resources/frameworks.json") as file:
    frameworks = json_load_as_obj(file)

print("Running `%s` on `%s` benchmarks in `%s` mode" % (args.framework, args.benchmark, args.mode))

framework = frameworks[args.framework]
Framework = namedtuple('Framework', framework._fields+('name',))
framework = Framework(**framework._asdict(), name=args.framework)

tasks = benchmarks[args.benchmark]

if args.mode == "local":
    bench = automl.Benchmark(framework, tasks)
elif args.mode == "docker":
    bench = automl.DockerBenchmark(framework, tasks, args.keep_instance)
elif args.mode == "aws":
    with open("resources/ami.json") as file:
        amis = json.load(file)

    ami = amis[args.region]
    if ami is None:
        raise ValueError("Region not supported by AMI yet.")

    bench = automl.AWSBenchmark(framework, tasks, ami, args.keep_instance)
else:
    raise ValueError("mode must be one of 'aws', 'docker' or 'local'.")

bench.setup()
if args.task is not None:
    res = bench.run_one(args.task, args.fold)
else:
    res = bench.run()
