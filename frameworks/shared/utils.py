from importlib import import_module
import importlib.util
import os
import sys


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_utils():
    amlb_path = os.environ.get("AMLB_PATH")
    if amlb_path:
        return load_module("amlb.utils", os.path.join(amlb_path, "utils", "__init__.py"))
    return import_module("amlb.utils")


utils = load_utils()
# unorthodox for it's only now that we can safely import those functions
from amlb.utils import *

