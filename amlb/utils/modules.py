import logging

try:
    from pip._internal import main as pip_main
except ImportError:
    from pip import main as pip_main


log = logging.getLogger(__name__)


def pip_install(module_or_requirements, is_requirements=False):
    try:
        if is_requirements:
            pip_main(['install', '--no-cache-dir', '-r', module_or_requirements])
        else:
            pip_main(['install', '--no-cache-dir', module_or_requirements])
    except SystemExit as se:
        log.error("Error when trying to install python modules %s.", module_or_requirements)
        log.exception(se)

