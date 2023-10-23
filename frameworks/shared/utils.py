from importlib import import_module
import importlib.util
import logging
import os
import pandas as pd
import sys


def setup_logger():
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    handlers = [console]
    logging.basicConfig(handlers=handlers)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    trace_level = os.environ.get('AMLB_LOG_TRACE')
    if trace_level:
        logging.TRACE = int(trace_level)


setup_logger()

__no_export = set(dir())  # all variables defined above this are not exported


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_amlb_module(mod, amlb_path=None):
    amlb_path = amlb_path or os.environ.get("AMLB_PATH")
    if amlb_path:
        tokens = mod.split('.')
        mod_path = os.path.join(amlb_path, *tokens)
        if os.path.isdir(mod_path):
            tokens.append("__init__.py")
            mod_path = os.path.join(amlb_path, *tokens)
        return load_module(mod, mod_path)
    return import_module(mod)


def load_timeseries_dataset(dataset):
    # Ensure that id_column is loaded as string to avoid incorrect sorting
    train_data = pd.read_csv(dataset.train_path, dtype={dataset.id_column: str}, parse_dates=[dataset.timestamp_column])
    test_data = pd.read_csv(dataset.test_path, dtype={dataset.id_column: str}, parse_dates=[dataset.timestamp_column])
    return train_data, test_data


utils = load_amlb_module("amlb.utils")
# unorthodox for it's only now that we can safely import those functions
from amlb.utils import *

__all__ = [s for s in dir() if not s.startswith('_') and s not in __no_export]
