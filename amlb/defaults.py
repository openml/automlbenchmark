import pathlib

import openml

from amlb.utils import Namespace as ns

# https://github.com/openml/automlbenchmark/pull/574#issuecomment-1646179921
try:
    cache_directory = openml.config.cache_directory
except AttributeError:
    cache_directory = openml.config.get_cache_directory()

default_dirs = ns(
    input_dir=cache_directory,
    output_dir=str(pathlib.Path(__file__).parent.parent / "results"),
    user_dir="~/.config/automlbenchmark",
    root_dir=str(pathlib.Path(__file__).parent.parent)
)
