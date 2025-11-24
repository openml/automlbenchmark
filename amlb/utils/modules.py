import logging
import sys

import subprocess

log = logging.getLogger(__name__)

__no_export = set(dir())  # all variables defined above this are not exported


def pip_install(module_or_requirements, is_requirements=False):
    try:
        cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir"]
        if is_requirements:
            cmd.extend(["-r", module_or_requirements])
        else:
            cmd.append(module_or_requirements)
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        log.error(
            "Error when trying to install python modules %s.", module_or_requirements
        )
        log.exception(e)


__all__ = [s for s in dir() if not s.startswith("_") and s not in __no_export]
