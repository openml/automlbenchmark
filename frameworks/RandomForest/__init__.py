from amlb.utils import call_in_subprocess


def run(*args, **kwargs):
    def exec_run(*args, **kwargs):
        # ensure import is done here to be able to modify the environment variables in the target script
        from .exec import run
        return run(*args, **kwargs)

    return call_in_subprocess(exec_run, *args, **kwargs)


__all__ = (run)
