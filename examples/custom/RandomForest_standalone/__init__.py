import os

from amlb.resources import config as rconfig
from amlb.utils import call_script_in_same_dir

__path__.append(os.path.join(rconfig().root_dir, 'frameworks', 'shared'))


def setup(*args, **kwargs):
    call_script_in_same_dir(__file__, "setup.sh", rconfig().root_dir, *args, **kwargs)


def run(*args, **kwargs):
    from .exec import run
    run(*args, **kwargs)


__all__ = (run)
