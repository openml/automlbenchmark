import datetime as dt
import functools as ft
import json
import logging
import os
import shutil
import stat

import psutil

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
        return getattr(self, item) if hasattr(self, item) else None

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

    def as_dict(self):
        return self.__dict__


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


def flatten(iterable):
    return ft.reduce(lambda l, r: (l.extend(r) if isinstance(r, list) else l.append(r)) or l, iterable, [])


def json_load(file, as_object=False):
    if as_object:
        return json.load(file, object_hook=lambda dic: Namespace(**dic))
    else:
        return json.load(file)


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
    if s.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    elif s.lower() in ('false', 'f', 'no', 'n', '0'):
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


def dir_of(caller_file, rel_to_project_root=False):
    abs_path = os.path.dirname(os.path.realpath(caller_file))
    if rel_to_project_root:
        project_root = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
        return os.path.relpath(abs_path, project_root)
    else:
        return abs_path


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


def run_cmd(cmd, return_output=True):
    # todo: switch to subprocess module (Popen) instead of os? would allow to use timeouts and kill signal
    #   besides, this implementation doesn't seem to work well with some commands if output is not read.
    output = None
    with os.popen(cmd) as subp:
        if return_output:
            output = subp.read()
    if subp.close():
        log.debug(output)
        output_tail = tail(output, 25) if output else 'Unknown Error'
        raise OSError("Error when running command `{cmd}`: {error}".format(cmd=cmd, error=output_tail))
    return output


def call_script_in_same_dir(caller_file, script_file):
    here = dir_of(caller_file)
    script = os.path.join(here, script_file)
    mod = os.stat(script).st_mode
    os.chmod(script, mod | stat.S_IEXEC)
    output = run_cmd(script)
    log.debug(output)
    return output


def available_memory_mb():
    return psutil.virtual_memory().available / (1 << 20)

