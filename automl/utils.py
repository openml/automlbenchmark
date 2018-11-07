from collections import namedtuple
import json
import logging
import os
import pathlib
import stat
from typing import Optional

import numpy as np
import psutil
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

try:
    from pip._internal import main as pip_main
except ImportError:
    from pip import main as pip_main


log = logging.getLogger(__name__)


def repr(obj):
    return "{clazz}: {attributes}".format(clazz=obj.__class__, attributes=obj.__dict__)


def cache(self, key, fn):
    """

    :param self: the object that will hold the cached value
    :param key: the key/attribute for the cached value
    :param fn: the function returning the value to be cached
    :return: the value returned by fn on first call
    """
    if not hasattr(self, key):
        value = fn(self)
        setattr(self, key, value)
    return getattr(self, key)


def cached(fn):
    """

    :param fn:
    :return:
    """
    result = '__cached_result__' + fn.__name__

    def decorator(self):
        return cache(self, result, fn)

    return decorator


def memoize(fn):
    memo = {}

    def decorator(self, key=None):
        if key not in memo:
            memo[key] = fn(self) if key is None else fn(self, key)
        return memo[key]

    return decorator


def lazy_property(prop_fn):
    """

    :param prop_fn:
    :return:
    """
    prop_name = '__cached__' + prop_fn.__name__

    @property
    def decorator(self):
        return cache(self, prop_name, prop_fn)

    return decorator


def dict_to_namedtuple(dic, type_name='Tuple'):
    return namedtuple(type_name, dic.keys())(*dic.values())


def merge_namedtuple(ntuple1, ntuple2, type_name=None):
    type_name = type(ntuple1).__name__ if not type_name else type_name
    return namedtuple(type_name, ntuple1._fields+ntuple2._fields)(**ntuple1._asdict(), **ntuple2._asdict())


def extend_namedtuple(ntuple, dict, type_name=None):
    return merge_namedtuple(ntuple, dict_to_namedtuple(dict), type_name)


def json_load(file, as_object=False):
    if as_object:
        return json.load(file, object_hook=lambda dic: dict_to_namedtuple(dic, 'JsonNode'))
    else:
        return json.load(file)


def pip_install(module_or_requirements, is_requirements=False):
    try:
        if is_requirements:
            pip_main(['install', '--no-cache-dir', '-r', module_or_requirements])
        else:
            pip_main(['install', '--no-cache-dir', module_or_requirements])
    except SystemExit as se:
        log.error("error when trying to install python modules %s", module_or_requirements)
        log.exception(se)


def dir_of(caller_file, rel_to_project_root=False):
    abs_path = os.path.dirname(os.path.realpath(caller_file))
    if rel_to_project_root:
        project_root = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
        return os.path.relpath(abs_path, project_root)
    else:
        return abs_path


def call_script_in_same_dir(caller_file, script_file):
    here = dir_of(caller_file)
    script = os.path.join(here, script_file)
    mod = os.stat(script).st_mode
    os.chmod(script, mod | stat.S_IEXEC)
    output = os.popen(script).read()
    log.debug(output)


def available_memory_mb():
    return psutil.virtual_memory().available / (1 << 20)


def encoder(values) -> Optional[LabelEncoder]:
    return LabelEncoder().fit(values) if values else None


def encode_labels(data, labels):
    """

    :param data:
    :param labels:
    :return:
    """
    le = LabelEncoder().fit(labels)
    return le.transform(data).reshape(-1, 1), le


def one_hot_encode_predictions(predictions, target_feature):
    """ Performs one-hot encoding on predictions, order of column depends on target.

    :param predictions: vector of target label predictions
    :param target_feature: the target feature with categorical values.
      This is used to order the columns of the one-hot encoding.
    :return: a one hot encoding of the class predictions as numpy array.
    """
    class_predictions_le = target_feature.encode(predictions).reshape(-1, 1)
    class_probabilities = OneHotEncoder().fit_transform(class_predictions_le)
    return class_probabilities.todense()


def save_predictions_to_file(class_probabilities, class_predictions, file_path):
    """ Save class probabilities and predicted labels to file in csv format.

    :param class_probabilities: (N,K)-matrix describing class probabilities.
    :param class_predictions:  (N,) or (N,1) vector.
    :param file_path: string. File to save the predictions to.
    :return: None
    """
    if class_predictions.ndim == 1:
        class_predictions = class_predictions.reshape(-1, 1)
    combined_predictions = np.hstack((class_probabilities, class_predictions)).astype(str)
    pathlib.Path(file_path).touch()
    np.savetxt(file_path, combined_predictions, delimiter=',', fmt="%s")
