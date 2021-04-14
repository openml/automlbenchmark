import os
from amlb.benchmark import __installed_file__
from amlb.utils import dir_of


def setup(*args, **kwargs):
    from sklearn import __version__
    with open(os.path.join(dir_of(__file__), __installed_file__), 'w') as f:
        f.write('\n'.join([__version__, ""]))


def run(*args, **kwargs):
    from .exec import run
    return run(*args, **kwargs)
