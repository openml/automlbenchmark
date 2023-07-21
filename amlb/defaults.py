import pathlib

from openml.config import get_cache_directory

from amlb.utils import Namespace as ns

default_dirs = ns(
    input_dir=get_cache_directory(),
    output_dir=str(pathlib.Path(__file__).parent.parent / "results"),
    user_dir="~/.config/automlbenchmark",
    root_dir=str(pathlib.Path(__file__).parent.parent)
)
