import argparse
import os

# prevent asap other modules from defining the root logger using basicConfig
import amlb.logger


import amlb
from amlb import log
from amlb.utils import Namespace as ns, config_load


parser = argparse.ArgumentParser()
parser.add_argument('instances', type=str, help="The path to an instances.csv file.")
parser.add_argument('-X', '--extra', default=[], action='append', help=argparse.SUPPRESS)
args = parser.parse_args()
extras = {t[0]: t[1] if len(t) > 1 else True for t in [x.split('=', 1) for x in args.extra]}

# script_name = os.path.splitext(os.path.basename(__file__))[0]
# log_dir = os.path.join(args.outdir if args.outdir else '.', 'logs')
# os.makedirs(log_dir, exist_ok=True)
# now_str = datetime_iso(date_sep='', time_sep='')
amlb.logger.setup(root_level='DEBUG', console_level='INFO')

config = config_load("resources/config.yaml")
config.run_mode = 'script'
config.script = os.path.basename(__file__)
amlb.resources.from_configs(config, ns.parse(extras))

aws = amlb.AWSBenchmark.fetch_results(args.instances, force_update=True)

