from amlb.utils import call_in_subprocess, call_script_in_same_dir, dir_of


def setup(*args):
    call_script_in_same_dir(__file__, "setup.sh", *args)


def run(*args, **kwargs):
    def exec_run(*args, **kwargs):
        # ensure import is done here to be able to modify the environment variables in the target script
        from .exec import run
        return run(*args, **kwargs)

    return call_in_subprocess(exec_run, *args, **kwargs)


def docker_commands(*args, **kwargs):
    return """
RUN {here}/setup.sh
""".format(here=dir_of(__file__, True))


__all__ = (setup, run, docker_commands)
