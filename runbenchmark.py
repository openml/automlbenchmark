import argparse

import automl
from automl.utils import json_load


parser = argparse.ArgumentParser()
parser.add_argument('framework', type=str,
                    help='The framework to evaluate as defined in resources/frameworks.json')
parser.add_argument('benchmark', type=str,
                    help='The benchmark type to run as defined in resources/benchmarks/{benchmark}.json or the path to a benchmark description file')
parser.add_argument('-m', '--mode', type=str, default='local',
                    help='The mode that specifies what backend is used (currently local [default], docker, or aws)')
parser.add_argument('-t', '--task', dest='task', metavar='-t', default=None,
                    help='The specific task name to run in the benchmark')
parser.add_argument('-f', '--fold', dest='fold', metavar='-f', type=int, default=0,
                    help='The specific fold to run in the benchmark')
parser.add_argument('-o', '--outdir', dest='outdir', metavar='-o', default=None,
                    help='Path where all the outputs should be written.')
parser.add_argument('-r', '--region', dest='region', metavar='-r', default=None,
                    help='The region on which to run the benchmark when using AWS.')
parser.add_argument('--reuse-instance', dest='reuse_instance', type=bool, default=False,
                    help='Set to true if reusing the same container instance(s) for all tasks (docker and aws mode only)')
args = parser.parse_args()

print("Running `%s` on `%s` benchmarks in `%s` mode" % (args.framework, args.benchmark, args.mode))

with open("resources/config.json") as file:
    config = json_load(file)

if args.outdir:
    config['output_folder'] = args.outdir

if args.mode == "local":
    bench = automl.Benchmark(args.framework, args.benchmark, config)
elif args.mode == "docker":
    bench = automl.DockerBenchmark(args.framework, args.benchmark, config, reuse_instance=args.reuse_instance)
elif args.mode == "aws":
    bench = automl.AWSBenchmark(args.framework, args.benchmark, config, region=args.region, reuse_instance=args.reuse_instance)
else:
    raise ValueError("mode must be one of 'aws', 'docker' or 'local'.")

bench.setup()
if args.task is not None:
    res = bench.run_one(args.task, args.fold)
else:
    res = bench.run()
