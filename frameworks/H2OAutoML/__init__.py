from amlb.utils import call_script_in_same_dir


def setup(*args, **kwargs):
    call_script_in_same_dir(__file__, "setup.sh", *args, **kwargs)


def run(*args, **kwargs):
    from .exec import run
    return run(*args, **kwargs)


def docker_commands(*args, setup_cmd=None):
    return """
{cmd}
EXPOSE 54321
EXPOSE 54322
""".format(
        cmd="RUN {}".format(setup_cmd) if setup_cmd is not None else ""
    )


#There is no network isolation in Singularity,
#so there is no need to map any port.
#If the process inside the container binds to an IP:port,
#it will be immediately reachable on the host.
def singularity_commands(*args, setup_cmd=None):
    return """
{cmd}
""".format(
        cmd="{}".format(setup_cmd) if setup_cmd is not None else ""
    )
