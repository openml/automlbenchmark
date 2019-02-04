from automl.utils import call_script_in_same_dir


def setup(*args):
    call_script_in_same_dir(__file__, "setup.sh", *args)


def run(*args, **kwargs):
    from .exec import run
    run(*args, **kwargs)


__all__ = (run)
