# prevent asap other modules from defining the root logger using basicConfig
import logging
logging.basicConfig(handlers=[logging.NullHandler()])

import argparse
import os

import automl
from automl.utils import json_load, datetime_iso, str2bool
from automl import log


parser = argparse.ArgumentParser()
parser.add_argument('framework', type=str,
                    help='The framework to evaluate as defined in resources/frameworks.json.')
parser.add_argument('benchmark', type=str, nargs='?', default='test',
                    help='The benchmark type to run as defined in resources/benchmarks/{benchmark}.json or the path to a benchmark description file. Defaults to `test`.')
parser.add_argument('-m', '--mode', choices=['local', 'docker', 'aws', 'aws-remote'], default='local',
                    help='The mode that specifies what backend is used (currently local [default], docker, or aws).')
parser.add_argument('-t', '--task', metavar='task_id', default=None,
                    help='The specific task name to run in the benchmark.')
parser.add_argument('-f', '--fold', metavar='fold_num', type=int, nargs='*',
                    help='The specific fold(s) to run in the benchmark.')
parser.add_argument('-i', '--indir', metavar='input_dir', default=None,
                    help='Folder where datasets are loaded by default.')
parser.add_argument('-o', '--outdir', metavar='output_dir', default=None,
                    help='Folder where all the outputs should be written.')
parser.add_argument('-r', '--region', metavar='aws_region', default=None,
                    help='The region on which to run the benchmark when using AWS.')
parser.add_argument('-s', '--setup', choices=['auto', 'skip', 'force', 'only'], default='auto',
                    help='Framework/platform setup mode (supported values = auto [default], skip, force, only).')
parser.add_argument('-p', '--parallel', metavar='jobs_num', type=int, default=1,
                    help='The number of jobs (i.e. tasks or folds) that can be run in parallel.'
                         'Currently supported only in aws mode.')
# parser.add_argument('--keep-instance', type=str2bool, metavar='true|false', nargs='?', const=True, default=True,
#                     help='Set to true [default] if reusing the same container instance(s) for all tasks (docker and aws mode only). '
#                          'If disabled in aws mode, we will try to distribute computing over multiple ec2 instances.')
# group = parser.add_mutually_exclusive_group()
# group.add_argument('--keep-instance', dest='keep_instance', action='store_true',
#                    help='Set to true [default] if reusing the same container instance(s) for all tasks (docker and aws mode only). '
#                         'If disabled in aws mode, we will try to distribute computing over multiple ec2 instances.')
# group.add_argument('--no-keep-instance', dest='keep_instance', action='store_false')
# parser.set_defaults(keep_instance=True)
args = parser.parse_args()

script_name = os.path.splitext(os.path.basename(__file__))[0]
log_dir = os.path.join(args.outdir if args.outdir else '.', 'logs')
os.makedirs(log_dir, exist_ok=True)
now_str = datetime_iso(date_sep='', time_sep='')
# now_str = now_iso(time=False, no_sep=True)
automl.logger.setup(log_file=os.path.join(log_dir, '{script}_{now}.log'.format(script=script_name, now=now_str)),
                    root_file=os.path.join(log_dir, '{script}_{now}_full.log'.format(script=script_name, now=now_str)),
                    root_level='DEBUG', console_level='INFO')

log.info("Running `%s` on `%s` benchmarks in `%s` mode", args.framework, args.benchmark, args.mode)
log.debug("script args: %s", args)

with open("resources/config.json") as file:
    # todo: allow a custom automlbenchmark_config.json in user directory: maybe this would allow removal of parameters like region, indir, outdir
    config = json_load(file, as_object=True)
    config.run_mode = args.mode
    config.script = os.path.basename(__file__)
    if args.indir:
        config.input_dir = args.indir
    if args.outdir:
        config.output_dir = args.outdir
automl.resources.from_config(config)

if args.mode == "local":
    bench = automl.Benchmark(args.framework, args.benchmark, parallel_jobs=args.parallel)
elif args.mode == "docker":
    bench = automl.DockerBenchmark(args.framework, args.benchmark, parallel_jobs=args.parallel)
elif args.mode == "aws":
    bench = automl.AWSBenchmark(args.framework, args.benchmark, parallel_jobs=args.parallel, region=args.region)
elif args.mode == "aws-remote":
    bench = automl.AWSRemoteBenchmark(args.framework, args.benchmark, parallel_jobs=args.parallel, region=args.region)
else:
    raise ValueError("mode must be one of 'aws', 'docker' or 'local'.")

if args.setup == 'only':
    log.warn("Setting up {} environment only, no benchmark will be run".format(args.mode))

bench.setup(automl.Benchmark.SetupMode[args.setup])
if args.setup != 'only':
    if args.task is None:
        res = bench.run(save_scores=True)
    else:
        res = bench.run_one(args.task, args.fold, save_scores=True)

