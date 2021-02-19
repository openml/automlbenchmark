from concurrent.futures import ThreadPoolExecutor
from functools import reduce, wraps
import inspect
import io
import logging
import multiprocessing as mp
import os
import queue
import re
import select
import signal
import stat
import subprocess
import sys
import threading
import _thread
import traceback

import psutil

from .core import Namespace, as_list, flatten, fn_name
from .os import dir_of, to_mb
from .time import Timeout, Timer

log = logging.getLogger(__name__)


def run_subprocess(*popenargs,
                   input=None, capture_output=False, timeout=None, check=False, communicate_fn=None,
                   **kwargs):
    """
    a clone of :function:`subprocess.run` which allows custom handling of communication
    :param popenargs:
    :param input:
    :param capture_output:
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

    if capture_output:
        if kwargs.get('stdout') is not None or kwargs.get('stderr') is not None:
            raise ValueError('stdout and stderr arguments may not be used '
                             'with capture_output.')
        kwargs['stdout'] = subprocess.PIPE
        kwargs['stderr'] = subprocess.PIPE

    def communicate(process, input=None, timeout=None):
        return (communicate_fn(process, input=input, timeout=timeout) if communicate_fn
                else process.communicate(input=input, timeout=timeout))

    with subprocess.Popen(*popenargs, **kwargs) as process:
        try:
            stdout, stderr = communicate(process, input, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            process.kill()
            if sys.platform == 'win32':
                e.stdout, e.stderr = communicate(process)
            else:
                process.wait()
            raise subprocess.TimeoutExpired(process.args, timeout, output=stdout, stderr=stderr)
        except:  # also handles kb interrupts
            process.kill()
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
        bufsize=-1,
        text=True,
        live_output=False,  # one of (True, 'line', 'block', False)
        output_level=logging.DEBUG,
        error_level=logging.ERROR,
        shell=True,
        executable=None,
        env=None,
        preexec_fn=None,
        timeout=None,
        activity_timeout=None,
        log_level=logging.INFO,
    )
    for k, v in params:
        kk = '_'+k+'_'
        if kk in kwargs:
            params[k] = kwargs[kk]
            del kwargs[kk]
    cmd_args = as_cmd_args(*args, **kwargs)
    full_cmd = flatten([cmd])+cmd_args
    str_cmd = ' '.join(full_cmd)
    log.log(params.log_level, "Running cmd `%s`", str_cmd)
    log.debug("Running cmd `%s` with input: %s", str_cmd, params.input_str)

    def live_output(process, input=None, **ignored):
        mode = params.live_output
        if mode is True:
            mode = 'line'

        if input is not None:
            try:
                with process.stdin as stream:
                    stream.write(input)
            except BrokenPipeError:
                pass
            except:
                raise

        def read_pipe(pipe, timeout):
            pipes = as_list(pipe)
            ready, *_ = select.select(pipes, [], [], timeout)
            reads = [''] * len(pipes)
            for i, p in enumerate(pipes):
                if p in ready:
                    line = p.readline()
                    if mode == 'line':
                        print(re.sub(r'\n$', '', line, count=1))
                    elif mode == 'block':
                        print(line, end='')
                    reads[i] = line
            return reads if len(pipes) > 1 else reads[0]

        output, error = zip(*iter(lambda: read_pipe([process.stdout if process.stdout else 1,
                                                     process.stderr if process.stderr else 2], params.activity_timeout),
                                  ['', '']))
        print()  # ensure that the log buffer is flushed at the end
        return ''.join(output), ''.join(error)

    try:
        completed = run_subprocess(str_cmd if params.shell else full_cmd,
                                   input=params.input_str,
                                   timeout=params.timeout,
                                   check=True,
                                   communicate_fn=live_output if params.live_output and params.capture_output else None,
                                   # stdin=subprocess.PIPE if params.input_str is not None else None,
                                   stdout=subprocess.PIPE if params.capture_output else None,
                                   stderr=subprocess.PIPE if params.capture_error else None,
                                   shell=params.shell,
                                   bufsize=params.bufsize,
                                   universal_newlines=params.text,
                                   executable=params.executable,
                                   env=params.env,
                                   preexec_fn=params.preexec_fn)
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


def run_script(script_path, *args, **kwargs):
    mod = os.stat(script_path).st_mode
    os.chmod(script_path, mod | stat.S_IEXEC)
    return run_cmd(script_path, *args, **kwargs)


def call_script_in_same_dir(caller_file, script_file, *args, **kwargs):
    here = dir_of(caller_file)
    script_path = os.path.join(here, script_file)
    return run_script(script_path, *args, **kwargs)


def get_thread(tid=None):
    return (threading.current_thread() if tid is None
            else threading.main_thread() if tid == 0
            else next(filter(lambda t: t.ident == tid, threading.enumerate()), None))


def get_process(pid=None):
    pid = (os.getpid() if pid is None
           else os.getppid() if pid == 0
           else pid)
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


def signal_handler(sig, handler):
    """
    :param sig: a signal as defined in https://docs.python.org/3.7/library/signal.html#module-contents
    :param handler: a handler function executed when the given signal is raised in the current thread.
    """
    prev_handler = None

    def handle(signum, frame):
        try:
            handler()
        finally:
            # restore previous signal handler
            signal.signal(sig, prev_handler or signal.SIG_DFL)

    prev_handler = signal.signal(sig, handle)


def raise_in_thread(thread_id, exc):
    """
    :param thread_id: the thread in which the exception will be raised.
    :param exc: the exception to raise in the thread: it can be an exception class or an instance.
    """
    import ctypes
    tid = ctypes.c_long(thread_id)
    exc_class = exc if inspect.isclass(exc) else type(exc.__class__.__name__, (exc.__class__,), dict(
        __init__=lambda s: super(s.__class__, s).__init__(str(exc))
    ))
    exc_class = ctypes.py_object(exc_class)
    ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, exc_class)
    if ret == 0:
        raise ValueError(f"Nonexistent thread {thread_id}")
    elif ret > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError(f"Failed raising exception in thread {thread_id}")


class InterruptTimeout(Timeout):
    """
    A :class:`Timeout` implementation that can send a signal to the interrupted thread or process,
    or raise an exception in the thread (works only for thread interruption)
    if the passed signal is an exception class or instance.
    """

    def __init__(self, timeout_secs, message=None, log_level=logging.WARNING,
                 interrupt='thread', sig=signal.SIGINT, ident=None, before_interrupt=None):
        def interruption():
            log.log(log_level, self.message)
            if before_interrupt is not None:
                before_interrupt()
            while not self.interrupt_event.is_set():
                try:
                    if interrupt == 'thread':
                        if isinstance(self.sig, (type(None), BaseException)):
                            exc = TimeoutError(self.message) if self.sig is None else self.sig
                            raise_in_thread(self.ident, exc)
                        else:
                            # _thread.interrupt_main()
                            signal.pthread_kill(self.ident, self.sig)
                    elif interrupt == 'process':
                        os.kill(self.ident, self.sig)
                except Exception:
                    raise
                finally:
                    self.interrupt_event.wait(1)  # retry every second if interruption didn't work

        super().__init__(timeout_secs, on_timeout=interruption)
        if interrupt not in ['thread', 'process']:
            raise ValueError("`interrupt` value should be one of ['thread', 'process'].")
        tp = get_thread(ident) if interrupt == 'thread' else get_process(ident)
        if tp is None:
            raise ValueError(f"no {interrupt} with id {ident}")
        if message is None:
            id = f"ident={tp.ident}" if isinstance(tp, threading.Thread) else f"pid={tp.pid}"
            self.message = f"Interrupting {interrupt} {tp.name} [{id}] after {timeout_secs}s timeout."
        else:
            self.message = message
        self.ident = tp.ident if tp is not None else None
        self.sig = sig(self.message) if inspect.isclass(sig) and BaseException in inspect.getmro(sig) else sig
        self.interrupt_event = threading.Event()

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        self.interrupt_event.set()
        if self.timed_out:
            if isinstance(self.sig, BaseException):
                raise self.sig
            elif self.sig is None:
                return True


class Monitoring:

    def __init__(self, name=None, frequency_seconds=300, check_on_exit=False, thread_prefix="monitoring_"):
        self._exec = None
        self._name = name or os.getpid()
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

    def __init__(self, name=None, frequency_seconds=300, check_on_exit=False,
                 use_interval=False, per_cpu=False, verbosity=0, log_level=logging.INFO):
        super().__init__(name=name,
                         frequency_seconds=0 if use_interval else frequency_seconds,
                         check_on_exit=check_on_exit,
                         thread_prefix="cpu_monitoring_")
        self._interval = frequency_seconds if use_interval else None
        self._per_cpu = per_cpu
        self._verbosity = verbosity
        self._log_level = log_level

    def _check_state(self):
        if self._verbosity == 0:
            percent = psutil.cpu_percent(interval=self._interval, percpu=self._per_cpu)
            log.log(self._log_level, "[%s] CPU Utilization: %s%%", self._name, percent)
        elif self._verbosity > 0:
            percent = psutil.cpu_times_percent(interval=self._interval, percpu=self._per_cpu)
            log.log(self._log_level, "[%s] CPU Utilization (in percent):\n%s", self._name, percent)


class MemoryMonitoring(Monitoring):

    def __init__(self, name=None, frequency_seconds=300, check_on_exit=False,
                 verbosity=0, log_level=logging.INFO):
        super().__init__(name=name,
                         frequency_seconds=frequency_seconds,
                         check_on_exit=check_on_exit,
                         thread_prefix="memory_monitoring_")
        self._verbosity = verbosity
        self._log_level = log_level

    def _check_state(self):
        if self._verbosity == 0:
            percent = system_memory_mb().used_percentage
            log.log(self._log_level, "[%s] Memory Usage: %s%%", self._name, percent)
        elif self._verbosity == 1:
            mem = system_memory_mb()
            log.log(self._log_level, "[%s] Memory Usage (in MB): %s", self._name, mem)
        elif self._verbosity > 1:
            mem = psutil.virtual_memory()
            log.log(self._log_level, "[%s] Memory Usage (in Bytes): %s",self._name,  mem)


class VolumeMonitoring(Monitoring):

    def __init__(self, name=None, frequency_seconds=300, check_on_exit=False, root="/",
                 verbosity=0, log_level=logging.INFO):
        super().__init__(name=name,
                         frequency_seconds=frequency_seconds,
                         check_on_exit=check_on_exit,
                         thread_prefix="volume_monitoring_")
        self._root = root
        self._verbosity = verbosity
        self._log_level = log_level

    def _check_state(self):
        if self._verbosity == 0:
            percent = system_volume_mb(self._root).used_percentage
            log.log(self._log_level, "[%s] Disk Usage: %s%%", self._name, percent)
        elif self._verbosity == 1:
            du = system_volume_mb(self._root)
            log.log(self._log_level, "[%s] Disk Usage (in MB): %s", self._name, du)
        elif self._verbosity > 1:
            du = psutil.disk_usage(self._root)
            log.log(self._log_level, "[%s] Disk Usage (in Bytes): %s", self._name, du)


class OSMonitoring(Monitoring):

    def __init__(self, name=None, frequency_seconds=300, check_on_exit=False,
                 statistics=('cpu', 'memory', 'volume'), verbosity=0, log_level=logging.INFO):
        super().__init__(name=name, frequency_seconds=frequency_seconds, check_on_exit=check_on_exit)
        self.monitors = []
        if 'cpu' in statistics:
            self.monitors.append(CPUMonitoring(name=name, frequency_seconds=frequency_seconds, verbosity=verbosity, log_level=log_level))
        if 'memory' in statistics:
            self.monitors.append(MemoryMonitoring(name=name, frequency_seconds=frequency_seconds, verbosity=verbosity, log_level=log_level))
        if 'volume' in statistics:
            self.monitors.append(VolumeMonitoring(name=name, frequency_seconds=frequency_seconds, verbosity=verbosity, log_level=log_level))

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



