import argparse
import os

# prevent asap other modules from defining the root logger using basicConfig
import amlb.logger

from ruamel import yaml

import amlb
from amlb import log
from amlb.utils import config_load

root_dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser()
parser.add_argument('predictions', type=str,
                    help='The predictions file to load and compute the scores for.')
args = parser.parse_args()

# script_name = os.path.splitext(os.path.basename(__file__))[0]
# log_dir = os.path.join(args.outdir if args.outdir else '.', 'logs')
# os.makedirs(log_dir, exist_ok=True)
# now_str = datetime_iso(date_sep='', time_sep='')
amlb.logger.setup(root_level='DEBUG', console_level='INFO')

config = config_load("resources/config.yaml")
config.run_mode = 'script'
config.root_dir = root_dir
config.script = os.path.basename(__file__)
amlb.resources.from_config(config)

scores = amlb.TaskResult.score_from_predictions_file(args.predictions)
log.info("\n\nScores computed from %s:\n%s", args.predictions, yaml.dump(dict(scores), default_flow_style=False))

