from automl.utils import call_script_in_same_dir


def setup(verbose=True):
    call_script_in_same_dir(__file__, "setup.sh", verbose)


def run(*args, **kwargs):
    from .exec import run
    run(*args, **kwargs)


__all__ = (setup, run)
