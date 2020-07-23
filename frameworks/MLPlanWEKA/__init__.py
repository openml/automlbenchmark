from amlb.utils import call_script_in_same_dir


def setup(*args, **kwargs):
    call_script_in_same_dir(__file__, "setup.sh", *args, **kwargs)


def run(*args, **kwargs):
    from .exec import run
    return run(*args, **kwargs)
