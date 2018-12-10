import datetime as dt
import functools as ft
import json
import logging
import os
import shutil
import stat

import psutil
from ruamel import yaml

try:
    from pip._internal import main as pip_main
except ImportError:
    from pip import main as pip_main


log = logging.getLogger(__name__)


class Namespace:

    mangled_prefix = '_Namespace__'

    @staticmethod
    def merge(*namespaces, deep=False):
        merged = Namespace()
        for ns in namespaces:
            if ns is None:
                continue
            if not deep:
                merged + ns
            else:
                for k, v in ns:
                    if isinstance(v, Namespace):
                        merged[k] = Namespace.merge(merged[k], v, deep=True)
                    else:
                        merged[k] = v
        return merged

    def __init__(self, *args, **kwargs):
        self.__ns = dict(*args, **kwargs)

    def __add__(self, other):
        self.__ns.update(other)
        return self

    def __contains__(self, key):
        return key in self.__ns

    def __len__(self):
        return len(self.__ns)

    def __getattr__(self, name):
        if name.startswith(Namespace.mangled_prefix):
            return super().__getattr__(name)
        elif name in self.__ns:
            return self.__ns[name]
        raise AttributeError(name)

    def __setattr__(self, key, value):
        if key.startswith(Namespace.mangled_prefix):
            super().__setattr__(key, value)
        else:
            self.__ns[key] = value

    def __getitem__(self, item):
        return self.__ns[item] if item in self.__ns else None

    def __setitem__(self, key, value):
        self.__ns[key] = value

    def __iter__(self):
        return iter(self.__ns.items())

    def __copy__(self):
        return Namespace(self.__ns.copy())

    def __dir__(self):
        return list(self.__ns.keys())

    def __str__(self):
        return str(self.__ns)

    def __repr__(self):
        return repr(self.__ns)


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

    def decorator(self, key=None):  # todo: could support unlimited args by making a tuple out of *args + **kwargs: not needed for now
        memo = cache(self, prop_name, lambda _: {})
        if not isinstance(key, str) and hasattr(key, '__iter__'):
            key = tuple(key)
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


def flatten(iterable, flatten_tuple=False, flatten_dict=False):
    return ft.reduce(lambda l, r: (l.extend(r) if isinstance(r, (list, tuple) if flatten_tuple else list)
                                   else l.extend(r.items()) if flatten_dict and isinstance(r, dict)
                                   else l.append(r)) or l, iterable, [])


class YAMLNamespaceLoader(yaml.loader.SafeLoader):

    @classmethod
    def init(cls):
        cls.add_constructor(u'tag:yaml.org,2002:map', cls.construct_yaml_map)

    def construct_yaml_map(self, node):
        data = Namespace()
        yield data
        value = self.construct_mapping(node)
        data + value


YAMLNamespaceLoader.init()


def json_load(file, as_namespace=False):
    if as_namespace:
        return json.load(file, object_hook=lambda dic: Namespace(**dic))
    else:
        return json.load(file)


def yaml_load(file, as_namespace=False):
    if as_namespace:
        return yaml.load(file, Loader=YAMLNamespaceLoader)
    else:
        return yaml.safe_load(file)


def config_load(path):
    path = normalize_path(path)
    if not os.path.isfile(path):
        log.warning("No config file at `%s`, skipping it.", path)
        return Namespace()

    base, ext = os.path.splitext(path.lower())
    loader = json_load if ext == 'json' else yaml_load
    log.info("Loading config file `%s`.", path)
    with open(path, 'r') as file:
        return loader(file, as_namespace=True)


def datetime_iso(datetime=None, date=True, time=True, micros=False, date_sep='-', datetime_sep='T', time_sep=':', micros_sep='.', no_sep=False):
    """

    :param date:
    :param time:
    :param micros:
    :param date_sep:
    :param time_sep:
    :param datetime_sep:
    :param micros_sep:
    :param no_sep: if True then all separators are taken as empty string
    :return:
    """
    # strf = "%Y{ds}%m{ds}%d{dts}%H{ts}%M{ts}%S{ms}%f".format(ds=date_sep, ts=time_sep, dts=datetime_sep, ms=micros_sep)
    if no_sep:
        date_sep = time_sep = datetime_sep = micros_sep = ''
    strf = ""
    if date:
        strf += "%Y{_}%m{_}%d".format(_=date_sep)
        if time:
            strf += datetime_sep
    if time:
        strf += "%H{_}%M{_}%S".format(_=time_sep)
        if micros:
            strf += "{_}%f".format(_=micros_sep)
    datetime = dt.datetime.utcnow() if datetime is None else datetime
    return datetime.strftime(strf)


def str2bool(s):
    if s.lower() in ('true', 't', 'yes', 'y', 'on', '1'):
        return True
    elif s.lower() in ('false', 'f', 'no', 'n', 'off', '0'):
        return False
    else:
        raise ValueError(s+" can't be interpreted as a boolean")


def str_def(s, if_none=''):
    if s is None:
        return if_none
    return str(s)


def head(s, lines=10):
    s_lines = s.splitlines() if s else []
    return '\n'.join(s_lines[:lines])


def tail(s, lines=10, from_line=None, include_line=True):
    if s is None:
        return None if from_line is None else None, None

    s_lines = s.splitlines()
    start = -lines
    if isinstance(from_line, int):
        start = from_line
        if not include_line:
            start += 1
    elif isinstance(from_line, str):
        try:
            start = s_lines.index(from_line)
            if not include_line:
                start += 1
        except ValueError:
            start = 0
    last_line = dict(index=len(s_lines) - 1,
                     line=s_lines[-1] if len(s_lines) > 0 else None)
    t = '\n'.join(s_lines[start:])
    return t if from_line is None else (t, last_line)


def pip_install(module_or_requirements, is_requirements=False):
    try:
        if is_requirements:
            pip_main(['install', '--no-cache-dir', '-r', module_or_requirements])
        else:
            pip_main(['install', '--no-cache-dir', module_or_requirements])
    except SystemExit as se:
        log.error("error when trying to install python modules %s", module_or_requirements)
        log.exception(se)


def normalize_path(path):
    return os.path.realpath(os.path.expanduser(path))


def dir_of(caller_file, rel_to_project_root=False):
    abs_path = os.path.dirname(os.path.realpath(caller_file))
    if rel_to_project_root:
        project_root = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
        return os.path.relpath(abs_path, project_root)
    else:
        return abs_path


def touch(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'a'):
        os.utime(file_path, times=None)


def backup_file(file_path):
    src_path = os.path.realpath(file_path)
    if not os.path.isfile(src_path):
        return
    dirname, basename = os.path.split(src_path)
    base, ext = os.path.splitext(basename)
    mod_time = dt.datetime.utcfromtimestamp(os.path.getmtime(src_path))
    dest_name = ''.join([base, '_', datetime_iso(mod_time, date_sep='', time_sep=''), ext])
    dest_dir = os.path.join(dirname, 'backup')
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, dest_name)
    shutil.copyfile(src_path, dest_path)
    log.info('file `%s` was backed up to `%s`.', src_path, dest_path)


def run_cmd(cmd, return_output=True, *args, **kvargs):
    # todo: switch to subprocess module (Popen) instead of os? would allow to use timeouts and kill signal
    #   besides, this implementation doesn't seem to work well with some commands if output is not read.
    output = None
    cmd_args = list(filter(None, []
                                 + ([] if args is None else list(args))
                                 + flatten(kvargs.items(), flatten_tuple=True) if kvargs is not None else []
                           ))
    full_cmd = ' '.join([cmd]+cmd_args)
    log.info("running cmd `%s`", full_cmd)
    with os.popen(full_cmd) as subp:
        if return_output:
            output = subp.read()
    if subp.close():
        log.debug(output)
        output_tail = tail(output, 25) if output else 'Unknown Error'
        raise OSError("Error when running command `{cmd}`: {error}".format(cmd=full_cmd, error=output_tail))
    return output


def call_script_in_same_dir(caller_file, script_file, *args, **kvargs):
    here = dir_of(caller_file)
    script = os.path.join(here, script_file)
    mod = os.stat(script).st_mode
    os.chmod(script, mod | stat.S_IEXEC)
    output = run_cmd(script, True, *args, **kvargs)
    log.debug(output)
    return output


def available_memory_mb():
    return psutil.virtual_memory().available / (1 << 20)

