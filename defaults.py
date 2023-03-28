import os

from openml.config import cache_directory

from amlb.utils import Namespace as ns

default_dirs = ns(
    input_dir=cache_directory,
    output_dir="./results",
    user_dir="~/.config/automlbenchmark",
    root_dir=os.path.dirname(__file__)
)
