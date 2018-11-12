import json
import logging
import os
import stat
from typing import Optional

import numpy as np
import psutil
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OneHotEncoder

try:
    from sklearn.preprocessing import OrdinalEncoder
except ImportError:
    from sklearn.preprocessing import LabelEncoder as OrdinalEncoder

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
    prop_name = '__memo__' + fn.__name__

    def decorator(self, key=None):
        memo = cache(self, prop_name, lambda _: {})
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


class Encoder(TransformerMixin):

    def __init__(self, type='label', target=True, encoded_type=float,
                 missing_handle='ignore', missing_values=None, missing_replaced_by=''):
        """

        :param type:
        :param target:
        :param missing_handle: one of ['ignore', 'mask', 'encode'].
            ignore: use only if there's no missing value for sure data to be transformed, otherwise it may raise an error during transform()
            mask: replace missing values only internally
            encode: encode all missing values as the encoded value of missing_replaced_by
        :param missing_values:
        :param missing_replaced_by:
        """
        super().__init__()
        assert missing_handle in ['ignore', 'mask', 'encode']
        self.for_target = target
        self.missing_handle = missing_handle
        self.missing_values = set(missing_values).union([None]) if missing_values else {None}
        self.missing_replaced_by = missing_replaced_by
        self.missing_encoded_value = None
        self.encoded_type = int if target else encoded_type
        self.str_encoder = None
        self.classes = None
        self._enc_classes_ = None
        if type == 'label':
            self.delegate = LabelEncoder() if target else OrdinalEncoder()
        elif type == 'one-hot':
            self.str_encoder = None if target else OrdinalEncoder()
            self.delegate = LabelBinarizer() if target else OneHotEncoder(handle_unknown='ignore')
        elif type == 'no-op':
            self.delegate = None
        else:
            raise ValueError("encoder type should be one of {}".format(['label', 'one-hot']))

    @property
    def _ignore_missing(self):
        return self.for_target or self.missing_handle == 'ignore'

    @property
    def _mask_missing(self):
        return not self.for_target and self.missing_handle == 'mask'

    @property
    def _encode_missing(self):
        return not self.for_target and self.missing_handle == 'encode'

    def fit(self, vec):
        if not self.delegate:
            return self

        self.classes = np.unique(vec) if self._ignore_missing else np.unique(np.insert(vec, 0, self.missing_replaced_by))
        self._enc_classes_ = self.str_encoder.fit_transform(self.classes) if self.str_encoder else self.classes

        if self._mask_missing:
            self.missing_encoded_value = self.delegate.fit_transform(self.classes)[0]
        else:
            self.delegate.fit(self.classes)
        return self

    def transform(self, vec, **params):
        if not self.delegate:
            return vec

        if self.str_encoder:
            vec = self.str_encoder.transform(vec)

        if self._mask_missing or self._encode_missing:
            mask = [v in self.missing_values for v in vec]
            if any(mask):
                nvec = vec if isinstance(vec, np.ndarray) else np.array(vec)
                if self._mask_missing:
                    missing = nvec[mask]
                nvec[mask] = self.missing_replaced_by
                res = self.delegate.transform(nvec, **params)
                if self._mask_missing and self.encoded_type != int:
                    if None in missing:
                        res = res.astype(self.encoded_type)
                    res[mask] = np.NaN if self.encoded_type == float else None
                return res

        return self.delegate.transform(vec, **params)

    def inverse_transform(self, vec, **params):
        if not self.delegate:
            return vec

        return self.delegate.inverse_transform(vec, **params)

