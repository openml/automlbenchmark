import argparse
import logging
import os
import re
import shutil
import sys

# prevent asap other modules from defining the root logger using basicConfig
import amlb.logger

import openml

import amlb
from amlb.utils import Namespace as ns, config_load, datetime_iso, str2bool, str_sanitize, zip_path
from amlb import log, AutoMLError
from amlb.defaults import default_dirs

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('framework', type=str,
                    help="The framework to evaluate as defined by default in resources/frameworks.yaml."
                         "\nTo use a labelled framework (i.e. a framework defined in resources/frameworks-{label}.yaml),"
                         "\nuse the syntax {framework}:{label}.")
parser.add_argument('benchmark', type=str, nargs='?', default='test',
                    help="The benchmark type to run as defined by default in resources/benchmarks/{benchmark}.yaml,"
                         "\na path to a benchmark description file, or an openml suite or task."
                         "\nOpenML references should be formatted as 'openml/s/X' and 'openml/t/Y',"
                         "\nfor studies and tasks respectively. Use 'test.openml/s/X' for the "
                         "\nOpenML test server."
                         "\n(default: '%(default)s')")
parser.add_argument('constraint', type=str, nargs='?', default='test',
                    help="The constraint definition to use as defined by default in resources/constraints.yaml."
                         "\n(default: '%(default)s')")
parser.add_argument('-m', '--mode', choices=['local', 'aws', 'docker', 'singularity'], default='local',
                    help="The mode that specifies how/where the benchmark tasks will be running."
                         "\n(default: '%(default)s')")
parser.add_argument('-t', '--task', metavar='task_id', nargs='*', default=None,
                    help="The specific task name (as defined in the benchmark file) to run."
                         "\nWhen an OpenML reference is used as benchmark, the dataset name should be used instead."
                         "\nIf not provided, then all tasks from the benchmark will be run.")
parser.add_argument('-f', '--fold', metavar='fold_num', type=int, nargs='*', default=None,
                    help="If task is provided, the specific fold(s) to run."
                         "\nIf fold is not provided, then all folds from the task definition will be run.")
parser.add_argument('-i', '--indir', metavar='input_dir', default=None,
                    help="Folder from where the datasets are loaded by default."
                         f"\n(default: '{default_dirs.input_dir}')")
parser.add_argument('-o', '--outdir', metavar='output_dir', default=None,
                    help="Folder where all the outputs should be written."
                         f"(default: '{default_dirs.output_dir}')")
parser.add_argument('-u', '--userdir', metavar='user_dir', default=None,
                    help="Folder where all the customizations are stored."
                         f"(default: '{default_dirs.user_dir}')")
parser.add_argument('--jobhistory', metavar='job_history', default=None,
                    help="File where prior job run results are stored. Only used when --resume is specified."
                         f"(default: 'None')")
parser.add_argument('-p', '--parallel', metavar='parallel_jobs', type=int, default=1,
                    help="The number of jobs (i.e. tasks or folds) that can run in parallel."
                         "\nA hard limit is defined by property `job_scheduler.max_parallel_jobs`"
                         "\n in `resources/config.yaml`."
                         "\nOverride this limit in your custom `config.yaml` file if needed."
                         "\nSupported only in aws mode or container mode (docker, singularity)."
                         "\n(default: %(default)s)")
parser.add_argument('-s', '--setup', choices=['auto', 'skip', 'force', 'only'], default='auto',
                    help="Framework/platform setup mode. Available values are:"
                         "\n• auto: setup is executed only if strictly necessary."
                         "\n• skip: setup is skipped."
                         "\n• force: setup is always executed before the benchmark."
                         "\n• only: only setup is executed (no benchmark)."
                         "\n(default: '%(default)s')")
parser.add_argument('-k', '--keep-scores', type=str2bool, metavar='true|false', nargs='?', const=True, default=True,
                    help="Set to true (default) to save/add scores in output directory.")
parser.add_argument('-e', '--exit-on-error', action='store_true', dest="exit_on_error",
                    help="If set, terminates on the first task that does not complete with a model.")
parser.add_argument('--logging', type=str, default="console:info,app:debug,root:info",
                    help="Set the log levels for the 3 available loggers:"
                         "\n• console"
                         "\n• app: for the log file including only logs from amlb (.log extension)."
                         "\n• root: for the log file including logs from libraries (.full.log extension)."
                         "\nAccepted values for each logger are: notset, debug, info, warning, error, fatal, critical."
                         "\nExamples:"
                         "\n  --logging=info (applies the same level to all loggers)"
                         "\n  --logging=root:debug (keeps defaults for non-specified loggers)" 
                         "\n  --logging=console:warning,app:info"
                         "\n(default: '%(default)s')")
parser.add_argument('--openml-test-server', type=str2bool, metavar='true|false', nargs='?', const=True, default=False,
                    help=argparse.SUPPRESS)  # "Set to true to connect to the OpenML test server instead."
parser.add_argument('--openml-run-tag', type=str, default=None,
                    help="Tag that will be saved in metadata and OpenML runs created during upload, must match '([a-zA-Z0-9_\-\.])+'.")

parser.add_argument('--profiling', nargs='?', const=True, default=False, help=argparse.SUPPRESS)
parser.add_argument('--resume', nargs='?', const=True, default=False, help=argparse.SUPPRESS)
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
log_dir = amlb.resources.output_dirs(args.outdir or default_dirs.output_dir,
                                     session=sid,
                                     subdirs='logs',
                                     create=True)['logs']
# now_str = datetime_iso(time=False, no_sep=True)
if args.profiling:
    logging.TRACE = logging.INFO
log_levels = ns({logger: int(level) if level.isnumeric() else level.upper()
                 for logger, level in [d.split(':') for d in args.logging.split(',')]} if ':' in args.logging
                else dict(console=args.logging.upper(), app=args.logging.upper(), root=args.logging.upper()) if args.logging
                else {}) | ns(console='INFO', app='DEBUG', root='INFO')  # adding defaults if needed
amlb.logger.setup(log_file=os.path.join(log_dir, '{script}.{now}.log'.format(script=script_name, now=now_str)),
                  root_file=os.path.join(log_dir, '{script}.{now}.full.log'.format(script=script_name, now=now_str)),
                  root_level=log_levels.root, app_level=log_levels.app, console_level=log_levels.console, print_to_log=True)

log.info("Running benchmark `%s` on `%s` framework in `%s` mode.", args.framework, args.benchmark, args.mode)
if args.openml_test_server:
    openml.config.start_using_configuration_for_example()
    log.info("Connecting to the OpenML test server.")

log.debug("Script args: %s.", args)

config_default = config_load(os.path.join(default_dirs.root_dir, "resources", "config.yaml"))
config_default_dirs = default_dirs
# allowing config override from user_dir: useful to define custom benchmarks and frameworks for example.
config_user = config_load(extras.get('config', os.path.join(args.userdir or default_dirs.user_dir, "config.yaml")))
# config listing properties set by command line
config_args = ns.parse(
    {'results.global_save': args.keep_scores},
    input_dir=args.indir,
    output_dir=args.outdir,
    user_dir=args.userdir,
    script=os.path.basename(__file__),
    run_mode=args.mode,
    parallel_jobs=args.parallel,
    sid=sid,
    exit_on_error=args.exit_on_error,
    test_server=args.openml_test_server,
    tag=args.openml_run_tag,
    command=' '.join(sys.argv),
) + ns.parse(extras)
if args.mode != 'local':
    config_args + ns.parse({'monitoring.frequency_seconds': 0})
config_args = ns({k: v for k, v in config_args if v is not None})
log.debug("Config args: %s.", config_args)
# merging all configuration files
amlb_res = amlb.resources.from_configs(config_default, config_default_dirs, config_user, config_args)
if args.resume:
    if args.jobhistory is not None:
        job_history = args.jobhistory
    else:
        job_history = os.path.join(amlb_res.config.output_dir, amlb.results.Scoreboard.results_file)
else:
    job_history = None

bench = None
exit_code = 0
try:
    bench_kwargs = dict(
        framework_name=args.framework,
        benchmark_name=args.benchmark,
        constraint_name=args.constraint,
    )
    if job_history is not None:
        bench_kwargs['job_history'] = job_history

    if args.mode == 'local':
        bench_cls = amlb.Benchmark
    elif args.mode == 'docker':
        bench_cls = amlb.DockerBenchmark
    elif args.mode == 'singularity':
        bench_cls = amlb.SingularityBenchmark
    elif args.mode == 'aws':
        bench_cls = amlb.AWSBenchmark
        # bench = amlb.AWSBenchmark(args.framework, args.benchmark, args.constraint, region=args.region)
    # elif args.mode == "aws-remote":
    #     bench = amlb.AWSRemoteBenchmark(args.framework, args.benchmark, args.constraint, region=args.region)
    else:
        raise ValueError("`mode` must be one of 'aws', 'docker', 'singularity' or 'local'.")
    bench = bench_cls(**bench_kwargs)

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
    archives = amlb.resources.config().archive
    if archives and bench:
        out_dirs = bench.output_dirs
        for d in archives:
            if d in out_dirs:
                zip_path(out_dirs[d], os.path.join(out_dirs.session, f"{d}.zip"), arc_path_format='long')
                shutil.rmtree(out_dirs[d], ignore_errors=True)
    if args.openml_test_server:
        openml.config.stop_using_configuration_for_example()

    sys.exit(exit_code)
