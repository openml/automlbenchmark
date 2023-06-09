import logging
import sys
import types

try:
    from pip._internal import main as pip_main
except ImportError:
    from pip import main as pip_main


log = logging.getLogger(__name__)

__no_export = set(dir())  # all variables defined above this are not exported


def register_module(module_name):
    if module_name not in sys.modules:
        mod = types.ModuleType(module_name)
        sys.modules[module_name] = mod
    return sys.modules[module_name]


def register_submodule(mod, name):
    fullname = '.'.join([mod.__name__, name])
    module = register_module(fullname)
    setattr(mod, name, module)


def pip_install(module_or_requirements, is_requirements=False):
    try:
        if is_requirements:
            pip_main(['install', '--no-cache-dir', '-r', module_or_requirements])
        else:
            pip_main(['install', '--no-cache-dir', module_or_requirements])
    except SystemExit as se:
        log.error("Error when trying to install python modules %s.", module_or_requirements)
        log.exception(se)


__all__ = [s for s in dir() if not s.startswith('_') and s not in __no_export]
