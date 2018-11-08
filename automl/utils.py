import json
import logging
import os
import stat
from typing import Optional

import psutil
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OneHotEncoder

try:
    from pip._internal import main as pip_main
except ImportError:
    from pip import main as pip_main


log = logging.getLogger(__name__)


class Namespace:

    def __init__(self, **kwargs):
        for name in kwargs:
            self[name] = kwargs[name]

    def __contains__(self, key):
        return key in self.__dict__

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __iter__(self):
        return iter(self.__dict__.items())

    def __add__(self, other):
        return Namespace(**self.__dict__).extend(other)

    def __repr__(self):
        return repr_def(self)

    def extend(self, namespace):
        for name, value in namespace:
            self[name] = value
        return self

    def clone(self):
        cloned = Namespace()
        return cloned.extend(self)


def repr_def(obj):
    return "{clazz}({attributes})".format(clazz=type(obj).__name__, attributes=', '.join(("{}={}".format(k, repr(v)) for k, v in obj.__dict__.items())))


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


def json_load(file, as_object=False):
    if as_object:
        return json.load(file, object_hook=lambda dic: Namespace(**dic))
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


def encoder(values, type='label') -> Optional[TransformerMixin]:
    if values is None:
        return
    if type == 'label':
        return LabelEncoder().fit(values)
    elif type == 'binary':
        return LabelBinarizer().fit(values)
    elif type == 'one_hot':
        return OneHotEncoder(handle_unknown='ignore').fit(values)
    else:
        raise ValueError("encoder type should be one of {}".format(['label', 'binary', 'one_hot']))
