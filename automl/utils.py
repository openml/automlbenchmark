"""
**utils** module provide a set of generic utility functions and decorators, which are not data-related
(data manipulation utility functions should go to **datautils**).

important
    This module can be imported by any other module (especially framework integration modules),
    therefore, it should have as few external dependencies as possible,
    and should have no dependency to any other **automl** module.
"""
from ast import literal_eval
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import fnmatch
from functools import reduce, wraps
import json
import logging
import multiprocessing as mp
import os
import pprint
import queue
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import threading
import _thread
import traceback

import psutil
from ruamel import yaml

try:
    from pip._internal import main as pip_main
except ImportError:
    from pip import main as pip_main


log = logging.getLogger(__name__)


""" CORE FUNCTIONS """


class Namespace:

    printer = pprint.PrettyPrinter(indent=2)

    @staticmethod
    def parse(*args, **kwargs):
        raw = dict(*args, **kwargs)
        parsed = Namespace()
        dots, nodots = partition(raw.keys(), lambda s: '.' in s)
        for k in nodots:
            v = raw[k]
            try:
                if isinstance(v, str):
                    v = literal_eval(v)
            except:
                pass
            parsed[k] = v
        sublevel = {}
        for k in dots:
            k1, k2 = k.split('.', 1)
            entry = [(k2, raw[k])]
            if k1 in sublevel:
                sublevel[k1].update(entry)
            else:
                sublevel[k1] = dict(entry)
        for k, v in sublevel.items():
            parsed[k] = Namespace.parse(v)
        return parsed

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

    @staticmethod
    def dict(namespace):
        dic = dict(namespace)
        for k, v in dic.items():
            if isinstance(v, Namespace):
                dic[k] = Namespace.dict(v)
        return dic

    def __init__(self, *args, **kwargs):
        self.__dict__.update(dict(*args, **kwargs))

    def __add__(self, other):
        """extends self with other (always overrides)"""
        if other is not None:
            self.__dict__.update(other)
        return self

    def __mod__(self, other):
        """extends self with other (adds only missing keys)"""
        if other is not None:
            for k, v in other:
                self.__dict__.setdefault(k, v)
        return self

    def __contains__(self, key):
        return key in self.__dict__

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, item):
        return self.__dict__.get(item)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __delitem__(self, key):
        self.__dict__.pop(key, None)

    def __iter__(self):
        return iter(self.__dict__.items())

    def __copy__(self):
        return Namespace(self.__dict__.copy())

    def __dir__(self):
        return list(self.__dict__.keys())

    def __str__(self):
        return Namespace.printer.pformat(Namespace.dict(self))

    def __repr__(self):
        return repr(self.__dict__)

    def __json__(self):
        return Namespace.dict(self)


def repr_def(obj):
    return "{clazz}({attributes})".format(clazz=type(obj).__name__, attributes=', '.join(("{}={}".format(k, repr(v)) for k, v in obj.__dict__.items())))


def to_mb(size_in_bytes):
    return size_in_bytes / (1 << 20)


def to_gb(size_in_bytes):
    return size_in_bytes / (1 << 30)


def noop():
    pass


def flatten(iterable, flatten_tuple=False, flatten_dict=False):
    return reduce(lambda l, r: (l.extend(r) if isinstance(r, (list, tuple) if flatten_tuple else list)
                                else l.extend(r.items()) if flatten_dict and isinstance(r, dict)
                                else l.append(r)) or l, iterable, [])


def partition(iterable, predicate=id):
    truthy, falsy = [], []
    for i in iterable:
        if predicate(i):
            truthy.append(i)
        else:
            falsy.append(i)
    return truthy, falsy


def translate_dict(dic, translation_dict):
    tr = dict()
    for k, v in dic.items():
        if k in translation_dict:
            tr[translation_dict[k]] = v
        else:
            tr[k] = v
    return tr


def str2bool(s):
    if s.lower() in ('true', 't', 'yes', 'y', 'on', '1'):
        return True
    elif s.lower() in ('false', 'f', 'no', 'n', 'off', '0'):
        return False
    else:
        raise ValueError(s+" can't be interpreted as a boolean.")


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


def fn_name(fn):
    return ".".join([fn.__module__, fn.__qualname__])


def json_load(file, as_namespace=False):
    with open(file, 'r') as f:
        return json_loads(f.read(), as_namespace=as_namespace)


def json_loads(s, as_namespace=False):
    if as_namespace:
        return json.loads(s, object_hook=lambda dic: Namespace(**dic))
    else:
        return json.loads(s)


def json_dump(o, file, style='default'):
    with open(file, 'w') as f:
        f.write(json_dumps(o, style=style))


def json_dumps(o, style='default'):
    """

    :param o:
    :param style: str among ('compact', 'default', 'pretty').
                - `compact` removes all blanks (no space, no newline).
                - `default` adds a space after each separator but prints on one line
                - `pretty` adds a space after each separator and indents after opening brackets.
    :return:
    """
    separators = (',', ':') if style == 'compact' else None
    indent = 4 if style == 'pretty' else None

    def default_encode(o):
        if hasattr(o, '__json__') and callable(o.__json__):
            return o.__json__()
        return json.encoder.JSONEncoder.default(None, o)

    return json.dumps(o, indent=indent, separators=separators, default=default_encode)


""" CONFIG """


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


def yaml_load(file, as_namespace=False):
    if as_namespace:
        return yaml.load(file, Loader=YAMLNamespaceLoader)
    else:
        return yaml.safe_load(file)


def config_load(path, verbose=False):
    path = normalize_path(path)
    if not os.path.isfile(path):
        log.log(logging.WARNING if verbose else logging.DEBUG, "No config file at `%s`, ignoring it.", path)
        return Namespace()

    _, ext = os.path.splitext(path.lower())
    loader = json_load if ext == 'json' else yaml_load
    log.log(logging.INFO if verbose else logging.DEBUG, "Loading config file `%s`.", path)
    with open(path, 'r') as file:
        return loader(file, as_namespace=True)


""" CACHING """


_CACHE_PROP_PREFIX_ = '__cached__'


def _cached_property_name(fn):
    return _CACHE_PROP_PREFIX_ + (fn.__name__ if hasattr(fn, '__name__') else str(fn))


def clear_cache(self, functions=None):
    cached_properties = [prop for prop in dir(self) if prop.startswith(_CACHE_PROP_PREFIX_)]
    properties_to_clear = cached_properties if functions is None \
        else [prop for prop in [_cached_property_name(fn) for fn in functions] if prop in cached_properties]
    for prop in properties_to_clear:
        delattr(self, prop)
    log.debug("Cleared cached properties: %s.", properties_to_clear)


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
    result = _cached_property_name(fn)

    def decorator(self):
        return cache(self, result, fn)

    return decorator


def memoize(fn):
    prop_name = _cached_property_name(fn)

    def decorator(self, key=None):  # TODO: could support unlimited args by making a tuple out of *args + **kwargs: not needed for now
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
    prop_name = _cached_property_name(prop_fn)

    @property
    def decorator(self):
        return cache(self, prop_name, prop_fn)

    return decorator


""" TIME """


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


class Timer:

    @staticmethod
    def _zero():
        return 0

    def __init__(self, clock=time.time, enabled=True):
        self.start = 0
        self.stop = 0
        self._time = clock if enabled else Timer._zero

    def __enter__(self):
        self.start = self._time()
        return self

    def __exit__(self, *args):
        self.stop = self._time()

    @property
    def duration(self):
        if self.stop > 0:
            return self.stop - self.start
        return self._time() - self.start


class Timeout:

    def __init__(self, timeout_secs, on_timeout=None):
        enabled = timeout_secs is not None and timeout_secs >= 0
        self.timer = threading.Timer(timeout_secs, on_timeout) if enabled else None

    @property
    def active(self):
        return self.timer and self.timer.is_alive()

    def __enter__(self):
        if self.timer:
            self.timer.start()
        return self

    def __exit__(self, *args):
        if self.timer:
            self.timer.cancel()


class InterruptTimeout(Timeout):

    def __init__(self, timeout_secs, message=None, log_level=logging.WARNING,
                 interrupt='thread', sig=signal.SIGINT, ident=None, before_interrupt=None):
        def interruption():
            nonlocal message
            if message is None:
                desc = 'current' if ident is None else 'main' if ident == 0 else self.ident
                message = "Interrupting {} {} after {}s timeout.".format(interrupt, desc, timeout_secs)
            log.log(log_level, message)
            if before_interrupt is not None:
                before_interrupt()
            if interrupt == 'thread':
                # _thread.interrupt_main()
                signal.pthread_kill(self.ident, sig)
            elif interrupt == 'process':
                os.kill(self.ident, sig)

        super().__init__(timeout_secs, on_timeout=interruption)
        if interrupt not in ['thread', 'process']:
            raise ValueError("`interrupt` value should be one of ['thread', 'process'].")
        self.ident = get_thread(ident).ident if interrupt == 'thread' else get_process(ident).pid


""" FILE SYSTEM """


def normalize_path(path):
    return os.path.realpath(os.path.expanduser(path))


def split_path(path):
    dir, file = os.path.split(path)
    base, ext = os.path.splitext(file)
    return Namespace(dirname=dir, filename=file, basename=base, extension=ext)


def path_from_split(split, real_path=True):
    return os.path.join(os.path.realpath(split.dirname) if real_path else split.dirname,
                        split.basename)+split.extension


def dir_of(caller_file, rel_to_project_root=False):
    abs_path = os.path.realpath(os.path.dirname(caller_file))
    if rel_to_project_root:
        project_root = os.path.normpath(os.path.join(os.path.realpath(os.path.dirname(__file__)), '..'))
        return os.path.relpath(abs_path, start=project_root)
    else:
        return abs_path


def list_all_files(paths, include=None, exclude=None):
    all_files = []
    paths = paths if isinstance(paths, list) else [paths]
    for path in paths:
        # path = normalize_path(path)
        if os.path.isdir(path):
            for root_dir, sub_dirs, files in os.walk(path):
                for name in files:
                    all_files.append(os.path.join(root_dir, name))
        elif os.path.isfile(path):
            all_files.append(path)
        else:
            log.warning("Skipping file `%s` as it doesn't exist.", path)

    if include is not None:
        include = include if isinstance(include, list) else [include]
        included = []
        for pattern in include:
            included.extend(fnmatch.filter(all_files, pattern))
        all_files = [file for file in all_files if file in included]

    if exclude is not None:
        exclude = exclude if isinstance(exclude, list) else [exclude]
        excluded = []
        for pattern in exclude:
            excluded.extend(fnmatch.filter(all_files, pattern))
        all_files = [file for file in all_files if file not in excluded]

    return all_files


def touch(path, as_dir=False):
    path = normalize_path(path)
    if not os.path.exists(path):
        dirname, basename = (path, '') if as_dir else os.path.split(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
        if basename:
            open(path, 'a').close()
    os.utime(path, times=None)


def backup_file(file_path):
    src_path = os.path.realpath(file_path)
    if not os.path.isfile(src_path):
        return
    p = split_path(src_path)
    mod_time = dt.datetime.utcfromtimestamp(os.path.getmtime(src_path))
    dest_name = ''.join([p.basename, '_', datetime_iso(mod_time, date_sep='', time_sep=''), p.extension])
    dest_dir = os.path.join(p.dirname, 'backup')
    touch(dest_dir, as_dir=True)
    dest_path = os.path.join(dest_dir, dest_name)
    shutil.copyfile(src_path, dest_path)
    log.debug('File `%s` was backed up to `%s`.', src_path, dest_path)


class TmpDir:

    def __init__(self):
        self.tmp_dir = None

    def __enter__(self):
        self.tmp_dir = tempfile.mkdtemp()
        return self.tmp_dir

    def __exit__(self, *args):
        shutil.rmtree(self.tmp_dir)
        # pass


""" PROCESS """


def run_subprocess(*popenargs, input=None, timeout=None, check=False, communicate_fn=None, **kwargs):
    """
    a clone of :function:`subprocess.run` which allows custom handling of communication
    :param popenargs:
    :param input:
    :param timeout:
    :param check:
    :param communicate_fn:
    :param kwargs:
    :return:
    """
    if input is not None:
        if 'stdin' in kwargs:
            raise ValueError('stdin and input arguments may not both be used.')
        kwargs['stdin'] = subprocess.PIPE

    def communicate(process, input=None, timeout=None):
        if communicate_fn:
            out, err = communicate_fn(process, input=input, timeout=timeout)
            process.wait()  # safety, in case not done by communicate_fn
            return out, err
        else:
            return process.communicate(input=input, timeout=timeout)

    with subprocess.Popen(*popenargs, **kwargs) as process:
        try:
            stdout, stderr = communicate(process, input, timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = communicate(process)
            raise subprocess.TimeoutExpired(process.args, timeout, output=stdout, stderr=stderr)
        except:
            process.kill()
            process.wait()
            raise
        retcode = process.poll()
        if check and retcode:
            raise subprocess.CalledProcessError(retcode, process.args, output=stdout, stderr=stderr)
    return subprocess.CompletedProcess(process.args, retcode, stdout, stderr)


def as_cmd_args(*args, **kwargs):
    return list(filter(None,
                       []
                       + ([] if args is None else list(args))
                       + flatten(kwargs.items(), flatten_tuple=True) if kwargs is not None else []
                       ))


def run_cmd(cmd, *args, **kwargs):
    params = Namespace(
        input_str=None,
        capture_output=True,
        capture_error=True,
        live_output=False,  # one of (True, 'line', 'block', False)
        output_level=logging.DEBUG,
        error_level=logging.ERROR,
        shell=True,
        timeout=None,
    )
    for k, v in params:
        kk = '_'+k+'_'
        if kk in kwargs:
            params[k] = kwargs[kk]
            del kwargs[kk]
    cmd_args = as_cmd_args(*args, **kwargs)
    full_cmd = flatten([cmd])+cmd_args
    str_cmd = ' '.join(full_cmd)
    log.info("Running cmd `%s`", str_cmd)
    log.debug("Running cmd `%s` with input: %s", str_cmd, params.input_str)

    def live_output(process, **ignored):
        mode = params.live_output
        if mode is True:
            mode = 'line'

        for line in iter(process.stdout.readline, ''):
            if mode == 'line':
                print(re.sub(r'\n$', '', line, count=1))
            elif mode == 'block':
                print(line, end='')
        print('\n')  # ensure that the log buffer is flushed at the end

    try:
        completed = run_subprocess(str_cmd if params.shell else full_cmd,
                                   input=params.input_str,
                                   timeout=params.timeout,
                                   check=True,
                                   communicate_fn=live_output if params.live_output and params.capture_output else None,
                                   # stdin=subprocess.PIPE if input is not None else None,
                                   stdout=subprocess.PIPE if params.capture_output else None,
                                   stderr=subprocess.PIPE if params.capture_error else None,
                                   shell=params.shell,
                                   universal_newlines=True)
        if completed.stdout:
            log.log(params.output_level, completed.stdout)
        if completed.stderr:
            log.log(params.error_level, completed.stderr)
        return completed.stdout, completed.stderr
    except subprocess.CalledProcessError as e:
        if e.stdout:
            log.log(params.output_level, e.stdout)
        if e.stderr:
            log.log(params.error_level, e.stderr)
        # error_tail = tail(e.stderr, 25) if e.stderr else 'Unknown Error'
        # raise subprocess.SubprocessError("Error when running command `{cmd}`: {error}".format(cmd=full_cmd, error=error_tail))
        raise e


def call_script_in_same_dir(caller_file, script_file, *args, **kwargs):
    here = dir_of(caller_file)
    script = os.path.join(here, script_file)
    mod = os.stat(script).st_mode
    os.chmod(script, mod | stat.S_IEXEC)
    return run_cmd(script, *args, **kwargs)


def get_thread(tid=None):
    return threading.current_thread() if tid is None \
        else threading.main_thread() if tid == 0 \
        else next(filter(lambda t: t.ident == tid, threading.enumerate()))


def get_process(pid=None):
    pid = os.getpid() if pid is None \
        else os.getppid() if pid == 0 \
        else pid
    return psutil.Process(pid) if psutil.pid_exists(pid) else None


def kill_proc_tree(pid=None, include_parent=True, timeout=None, on_terminate=None):
    def on_proc_terminated(proc):
        log.info("Process %s terminated with exit code %s", proc, proc.returncode)
        if on_terminate is not None:
            on_terminate(proc)

    parent = get_process(pid)
    children = parent.children(recursive=True)
    if include_parent:
        children.append(parent)
    for proc in children:
        log.warning("Terminating process %s.", proc)
        proc.terminate()
    terminated, alive = psutil.wait_procs(children, timeout=timeout, callback=on_proc_terminated)
    for proc in alive:
        log.warning("Killing process %s.", proc)
        proc.kill()


def call_in_subprocess(target, *args, **kwargs):
    def call_target(q, *args, **kwargs):
        try:
            result = target(*args, **kwargs)
            q.put_nowait(result)
        except BaseException as e:
            log.exception(e)
            q.put_nowait(e)

    q = mp.Queue(maxsize=1)
    p = mp.Process(target=call_target, args=(q, *args), kwargs=kwargs)
    try:
        p.start()
        p.join()
        result = q.get_nowait()
        if isinstance(result, BaseException):
            raise result
        else:
            return result
    except queue.Empty:
        raise Exception("Subprocess running {} died abruptly.".format(target.__name__))
    except BaseException:
        try:
            kill_proc_tree(p.pid)
        except:
            pass
        raise


def system_cores():
    return psutil.cpu_count()


def system_memory_mb():
    vm = psutil.virtual_memory()
    return Namespace(
        total=to_mb(vm.total),
        available=to_mb(vm.available),
        used_percentage=vm.percent
    )


def system_volume_mb(root="/"):
    du = psutil.disk_usage(root)
    return Namespace(
        total=to_mb(du.total),
        free=to_mb(du.free),
        used=to_mb(du.used),
        used_percentage=du.percent
    )


class Monitoring:

    def __init__(self, frequency_seconds=300, check_on_exit=False, thread_prefix="monitoring_"):
        self._exec = None
        self._frequency = frequency_seconds
        self._thread_prefix = thread_prefix
        self._interrupt = threading.Event()
        self._check_on_exit = check_on_exit

    def __enter__(self):
        if self._frequency > 0:
            self._interrupt.clear()
            self._exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix=self._thread_prefix)
            self._exec.submit(self._monitor)
        return self

    def __exit__(self, *args):
        if self._exec is not None:
            self._interrupt.set()
            self._exec.shutdown(wait=False)
            if self._check_on_exit:
                self._check_state()
            self._exec = None

    def _monitor(self):
        while not self._interrupt.is_set():
            try:
                self._check_state()
            except Exception as e:
                log.exception(e)
            finally:
                self._interrupt.wait(self._frequency)

    def _check_state(self):
        pass


class CPUMonitoring(Monitoring):

    def __init__(self, frequency_seconds=300, check_on_exit=False,
                 use_interval=False, per_cpu=False, verbosity=0, log_level=logging.INFO):
        super().__init__(frequency_seconds=0 if use_interval else frequency_seconds,
                         check_on_exit=check_on_exit,
                         thread_prefix="cpu_monitoring_")
        self._interval = frequency_seconds if use_interval else None
        self._per_cpu = per_cpu
        self._verbosity = verbosity
        self._log_level = log_level

    def _check_state(self):
        if self._verbosity == 0:
            percent = psutil.cpu_percent(interval=self._interval, percpu=self._per_cpu)
            log.log(self._log_level, "CPU Utilization: %s%%", percent)
        elif self._verbosity > 0:
            percent = psutil.cpu_times_percent(interval=self._interval, percpu=self._per_cpu)
            log.log(self._log_level, "CPU Utilization (in percent):\n%s", percent)


class MemoryMonitoring(Monitoring):

    def __init__(self, frequency_seconds=300, check_on_exit=False,
                 verbosity=0, log_level=logging.INFO):
        super().__init__(frequency_seconds=frequency_seconds,
                         check_on_exit=check_on_exit,
                         thread_prefix="memory_monitoring_")
        self._verbosity = verbosity
        self._log_level = log_level

    def _check_state(self):
        if self._verbosity == 0:
            percent = system_memory_mb().used_percentage
            log.log(self._log_level, "Memory Usage: %s%%", percent)
        elif self._verbosity == 1:
            mem = system_memory_mb()
            log.log(self._log_level, "Memory Usage (in MB): %s", mem)
        elif self._verbosity > 1:
            mem = psutil.virtual_memory()
            log.log(self._log_level, "Memory Usage (in Bytes): %s", mem)


class VolumeMonitoring(Monitoring):

    def __init__(self, frequency_seconds=300, check_on_exit=False, root="/",
                 verbosity=0, log_level=logging.INFO):
        super().__init__(frequency_seconds=frequency_seconds,
                         check_on_exit=check_on_exit,
                         thread_prefix="volume_monitoring_")
        self._root = root
        self._verbosity = verbosity
        self._log_level = log_level

    def _check_state(self):
        if self._verbosity == 0:
            percent = system_volume_mb(self._root).used_percentage
            log.log(self._log_level, "Disk Usage: %s%%", percent)
        elif self._verbosity == 1:
            du = system_volume_mb(self._root)
            log.log(self._log_level, "Disk Usage (in MB): %s", du)
        elif self._verbosity > 1:
            du = psutil.disk_usage(self._root)
            log.log(self._log_level, "Disk Usage (in Bytes): %s", du)


class OSMonitoring(Monitoring):

    def __init__(self, frequency_seconds=300, check_on_exit=False,
                 statistics=('cpu', 'memory', 'volume'), verbosity=0, log_level=logging.INFO):
        super().__init__(frequency_seconds=frequency_seconds, check_on_exit=check_on_exit)
        self.monitors = []
        if 'cpu' in statistics:
            self.monitors.append(CPUMonitoring(frequency_seconds=frequency_seconds, verbosity=verbosity, log_level=log_level))
        if 'memory' in statistics:
            self.monitors.append(MemoryMonitoring(frequency_seconds=frequency_seconds, verbosity=verbosity, log_level=log_level))
        if 'volume' in statistics:
            self.monitors.append(VolumeMonitoring(frequency_seconds=frequency_seconds, verbosity=verbosity, log_level=log_level))

    def _check_state(self):
        for monitor in self.monitors:
            monitor._check_state()


class MemoryProfiler:

    def __init__(self, process=psutil.Process(), enabled=True):
        self.ps = process if enabled else None
        self.before_mem = None
        self.after_mem = None

    def __enter__(self):
        if self.ps is not None:
            self.before_mem = self.ps.memory_full_info()
        return self

    def __exit__(self, *args):
        if self.ps is not None:
            self.after_mem = self.ps.memory_full_info()

    def usage(self):
        if self.ps is not None:
            mem = self.after_mem if self.after_mem is not None else self.ps.memory_full_info()
            return Namespace(
                process_diff=to_mb(mem.uss-self.before_mem.uss),
                process=to_mb(mem.uss),
                resident_diff=to_mb(mem.rss-self.before_mem.rss),
                resident=to_mb(mem.rss),
                virtual_diff=to_mb(mem.vms-self.before_mem.vms),
                virtual=to_mb(mem.vms)
            )


def obj_size(o):
    if o is None:
        return 0
    # handling numpy obj size (nbytes property)
    return o.nbytes if hasattr(o, 'nbytes') else sys.getsizeof(o, -1)


def profile(logger=log, log_level=None, duration=True, memory=True):
    def decorator(fn):

        @wraps(fn)
        def profiler(*args, **kwargs):
            nonlocal log_level
            log_level = log_level or (logging.TRACE if hasattr(logging, 'TRACE') else logging.DEBUG)
            if not logger.isEnabledFor(log_level):
                return fn(*args, **kwargs)

            with Timer(enabled=duration) as t, MemoryProfiler(enabled=memory) as m:
                ret = fn(*args, **kwargs)
            name = fn_name(fn)
            if duration:
                logger.log(log_level, "[PROFILING] `%s` executed in %.3fs.", name, t.duration)
            if memory:
                ret_size = obj_size(ret)
                if ret_size > 0:
                    logger.log(log_level, "[PROFILING] `%s` returned object size: %.3f MB.", name, to_mb(ret_size))
                mem = m.usage()
                logger.log(log_level, "[PROFILING] `%s` memory change; process: %+.2f MB/%.2f MB, resident: %+.2f MB/%.2f MB, virtual: %+.2f MB/%.2f MB.",
                           name, mem.process_diff, mem.process, mem.resident_diff, mem.resident, mem.virtual_diff, mem.virtual)
            return ret

        return profiler

    return decorator


""" MODULES """


def pip_install(module_or_requirements, is_requirements=False):
    try:
        if is_requirements:
            pip_main(['install', '--no-cache-dir', '-r', module_or_requirements])
        else:
            pip_main(['install', '--no-cache-dir', module_or_requirements])
    except SystemExit as se:
        log.error("Error when trying to install python modules %s.", module_or_requirements)
        log.exception(se)

