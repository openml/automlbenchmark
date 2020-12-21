import argparse
import logging
import os
import re
import sys

# prevent asap other modules from defining the root logger using basicConfig
import openml

import amlb.logger

import amlb
from amlb.utils import Namespace as ns, config_load, datetime_iso, str2bool, str_sanitize
from amlb import log, AutoMLError


parser = argparse.ArgumentParser()
parser.add_argument('framework', type=str,
                    help="The framework to evaluate as defined by default in resources/frameworks.yaml. "
                         "To use a labelled framework (i.e. a framework defined in resources/frameworks-{label}.yaml), "
                         "use the syntax {framework}:{label}.")
parser.add_argument('benchmark', type=str, nargs='?', default='test',
                    help="The benchmark type to run as defined by default in resources/benchmarks/{benchmark}.yaml, "
                         "a path to a benchmark description file, or an openml suite or task. OpenML references should "
                         "be formatted as 'openml/s/X' and 'openml/t/Y', for studies and tasks respectively. Defaults to `%(default)s`.")
parser.add_argument('constraint', type=str, nargs='?', default='test',
                    help="The constraint definition to use as defined by default in resources/constraints.yaml. Defaults to `test`.")
parser.add_argument('-m', '--mode', choices=['local', 'aws', 'docker', 'singularity'], default='local',
                    help="The mode that specifies how/where the benchmark tasks will be running. Defaults to %(default)s.")
parser.add_argument('-t', '--task', metavar='task_id', nargs='*', default=None,
                    help="The specific task name (as defined in the benchmark file) to run. "
                         "When an OpenML reference is used as benchmark, the dataset name should be used instead. "
                         "If not provided, then all tasks from the benchmark will be run.")
parser.add_argument('-f', '--fold', metavar='fold_num', type=int, nargs='*', default=None,
                    help="If task is provided, the specific fold(s) to run. "
                         "If fold is not provided, then all folds from the task definition will be run.")
parser.add_argument('-i', '--indir', metavar='input_dir', default=None,
                    help="Folder where datasets are loaded by default. Defaults to `input_dir` as defined in resources/config.yaml")
parser.add_argument('-o', '--outdir', metavar='output_dir', default=None,
                    help="Folder where all the outputs should be written. Defaults to `output_dir` as defined in resources/config.yaml")
parser.add_argument('-u', '--userdir', metavar='user_dir', default=None,
                    help="Folder where all the customizations are stored. Defaults to `user_dir` as defined in resources/config.yaml")
parser.add_argument('-p', '--parallel', metavar='parallel_jobs', type=int, default=1,
                    help="The number of jobs (i.e. tasks or folds) that can run in parallel. Defaults to %(default)s. "
                         "Currently supported only in docker and aws mode.")
parser.add_argument('-s', '--setup', choices=['auto', 'skip', 'force', 'only'], default='auto',
                    help="Framework/platform setup mode. Defaults to %(default)s. "
                         "•auto: setup is executed only if strictly necessary. •skip: setup is skipped. •force: setup is always executed before the benchmark. •only: only setup is executed (no benchmark).")
parser.add_argument('-k', '--keep-scores', type=str2bool, metavar='true|false', nargs='?', const=True, default=True,
                    help="Set to true [default] to save/add scores in output directory.")
parser.add_argument('--tag', type=str, default=None,
                    help="Tag that will be saved in metadata and OpenML runs created during upload.")
parser.add_argument('--test-server', type=str2bool, metavar='true|false', nargs='?', const=True, default=False,
                    help=argparse.SUPPRESS)  # "Set to true to connect to the OpenML test server instead."
parser.add_argument('--profiling', nargs='?', const=True, default=False, help=argparse.SUPPRESS)
parser.add_argument('--session', type=str, default=None, help=argparse.SUPPRESS)
parser.add_argument('-X', '--extra', default=[], action='append', help=argparse.SUPPRESS)
# group = parser.add_mutually_exclusive_group()
# group.add_argument('--keep-scores', dest='keep_scores', action='store_true',
#                    help="Set to true [default] to save/add scores in output directory")
# group.add_argument('--no-keep-scores', dest='keep_scores', action='store_false')
# parser.set_defaults(keep_scores=True)

# removing this command line argument for now: by default, we're using the user default region as defined in ~/aws/config
#  on top of this, user can now override the aws.region setting in his custom ~/.config/automlbenchmark/config.yaml settings.
# parser.add_argument('-r', '--region', metavar='aws_region', default=None,
#                     help="The region on which to run the benchmark when using AWS.")

root_dir = os.path.dirname(__file__)
args = parser.parse_args()
script_name = os.path.splitext(os.path.basename(__file__))[0]
extras = {t[0]: t[1] if len(t) > 1 else True for t in [x.split('=', 1) for x in args.extra]}

now_str = datetime_iso(date_sep='', time_sep='')
sid = (args.session if args.session is not None
       else "{}.{}".format('.'.join([str_sanitize(args.framework.split(':', 1)[0]),
                                     str_sanitize(args.benchmark if re.fullmatch(r"(openml)/[st]/\d+", args.benchmark)
                                                  else os.path.splitext(os.path.basename(args.benchmark))[0]),
                                     str_sanitize(args.constraint),
                                     extras.get('run_mode', args.mode)])
                              .lower(),
                           now_str))
log_dir = amlb.resources.output_dirs(args.outdir or os.path.join(os.getcwd(), 'logs'),
                                     session=sid,
                                     subdirs='logs' if args.outdir else '',
                                     create=True)['logs' if args.outdir else 'session']
# now_str = datetime_iso(time=False, no_sep=True)
if args.profiling:
    logging.TRACE = logging.INFO
amlb.logger.setup(log_file=os.path.join(log_dir, '{script}.{now}.log'.format(script=script_name, now=now_str)),
                  root_file=os.path.join(log_dir, '{script}.{now}.full.log'.format(script=script_name, now=now_str)),
                  root_level='INFO', app_level='DEBUG', console_level='INFO', print_to_log=True)

log.info("Running `%s` on `%s` benchmarks in `%s` mode.", args.framework, args.benchmark, args.mode)
if args.test_server:
    openml.config.start_using_configuration_for_example()
    log.info("Connecting to the OpenML test server.")
log.debug("Script args: %s.", args)

config = config_load(os.path.join(root_dir, "resources", "config.yaml"))
# allowing config override from user_dir: useful to define custom benchmarks and frameworks for example.
config_user = config_load(os.path.join(args.userdir if args.userdir is not None else config.user_dir, "config.yaml"))
# config listing properties set by command line
config_args = ns.parse(
    {'results.save': args.keep_scores},
    input_dir=args.indir,
    output_dir=args.outdir,
    user_dir=args.userdir,
    root_dir=root_dir,
    script=os.path.basename(__file__),
    run_mode=args.mode,
    parallel_jobs=args.parallel,
    sid=sid,
    test_server=args.test_server,
    tag=args.tag,
    command=' '.join(sys.argv),
) + ns.parse(extras)
if args.mode != 'local':
    config_args + ns.parse({'monitoring.frequency_seconds': 0})
config_args = ns({k: v for k, v in config_args if v is not None})
log.debug("Config args: %s.", config_args)
# merging all configuration files
amlb.resources.from_configs(config, config_user, config_args)

exit_code = 0
try:
    if args.mode == 'local':
        bench = amlb.Benchmark(args.framework, args.benchmark, args.constraint)
    elif args.mode == 'docker':
        bench = amlb.DockerBenchmark(args.framework, args.benchmark, args.constraint)
    elif args.mode == 'singularity':
        bench = amlb.SingularityBenchmark(args.framework, args.benchmark, args.constraint)
    elif args.mode == 'aws':
        bench = amlb.AWSBenchmark(args.framework, args.benchmark, args.constraint)
        # bench = amlb.AWSBenchmark(args.framework, args.benchmark, args.constraint, region=args.region)
    # elif args.mode == "aws-remote":
    #     bench = amlb.AWSRemoteBenchmark(args.framework, args.benchmark, args.constraint, region=args.region)
    else:
        raise ValueError("`mode` must be one of 'aws', 'docker', 'singularity' or 'local'.")

    if args.setup == 'only':
        log.warning("Setting up %s environment only for %s, no benchmark will be run.", args.mode, args.framework)

    if not args.keep_scores and args.mode != 'local':
        log.warning("`keep_scores` parameter is currently ignored in %s mode, scores are always saved in this mode.", args.mode)

    bench.setup(amlb.SetupMode[args.setup])
    if args.setup != 'only':
        res = bench.run(args.task, args.fold)
except (ValueError, AutoMLError) as e:
    log.error('\nERROR:\n%s', e)
    if extras.get('verbose') is True:
        log.exception(e)
    exit_code = 1
except Exception as e:
    log.exception(e)
    exit_code = 2
finally:
    if args.test_server:
        openml.config.stop_using_configuration_for_example()

sys.exit(exit_code)
