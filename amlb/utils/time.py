from __future__ import annotations

import datetime as dt
import logging
import math
import threading
import time
from typing import Callable

from .core import identity, threadsafe_generator

log = logging.getLogger(__name__)

__no_export = set(dir())  # all variables defined above this are not exported


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


def countdown(timeout_secs, on_timeout: Callable | None = None, message: str = "", interval=1, log_level=logging.INFO,
              interrupt_event: threading.Event | None = None, interrupt_cond: Callable | None = None):
    timeout_epoch = time.time() + timeout_secs
    remaining = timeout_secs
    interrupt = interrupt_event or threading.Event()
    while remaining > 0 and not interrupt.is_set():
        mins, secs = divmod(remaining, 60)
        hours, mins = divmod(mins, 60)
        if message:
            log.log(log_level, "in %02d:%02d:%02d : %s", hours, mins, secs, message)
        else:
            log.log(log_level, "countdown: %02d:%02d:%02d", hours, mins, secs)
        next_sleep = min(interval, remaining)
        interrupt.wait(next_sleep)
        remaining = math.ceil(timeout_epoch - time.time())
        if not interrupt.is_set() and interrupt_cond and interrupt_cond():
            interrupt.set()
    if on_timeout:
        on_timeout()


@threadsafe_generator
def retry_after(start=0, fn=identity, max_retries=math.inf):
    """
    generator returning a delay (usually interpreted as seconds) before next retry
    :param start: the first delay
    :param fn: the function computing the next delay from the previous one
    :param max_retries:
    :return:
    """
    delay = start
    retries = 1
    while True:
        if 0 <= max_retries < retries:
            return
        yield delay
        retries = retries+1
        delay = fn(delay)


def retry_policy(policy: str):
    tokens = policy.split(':', 3)
    type = tokens[0]
    l = len(tokens)
    if type == 'constant':
        interval = float(tokens[2] if l > 2 else tokens[1] if l > 1 else 60)
        start = float(tokens[1] if l > 2 else interval)
        return start, (lambda _: interval)
    elif type == 'linear':
        max_delay = float(tokens[3] if l > 3 else math.inf)
        increment = float(tokens[2] if l > 2 else tokens[1] if l > 1 else 60)
        start = float(tokens[1] if l > 2 else increment)
        return start, (lambda d: min(d + increment, max_delay))
    elif type == 'exponential':
        max_delay = float(tokens[3] if l > 3 else math.inf)
        factor = float(tokens[2] if l > 2 else tokens[1] if l > 1 else 2)
        start = float(tokens[1] if l > 2 else 60)
        return start, (lambda d: min(d * factor, max_delay))
    else:
        raise ValueError(f"Unsupported policy {type} in '{policy}': supported policies are [constant, linear, exponential].")


class Timer:

    @staticmethod
    def _zero():
        return 0

    def __init__(self, clock=time.time, enabled=True):
        self.start = 0
        self.stop = 0
        self._time = clock if enabled else Timer._zero
        self._tick = 0

    def __enter__(self):
        self.start = self._tick = self._time()
        return self

    def __exit__(self, *args):
        self.stop = self._tick = self._time()

    @property
    def tick(self):
        if self.stop > 0:
            return -1
        now = self._time()
        tick = now - self._tick
        self._tick = now
        return tick

    @property
    def duration(self):
        if self.stop > 0:
            return self.stop - self.start
        return self._time() - self.start


class Timeout:

    def __init__(self, timeout_secs, on_timeout=None):
        def timeout_handler():
            self.timed_out = True
            on_timeout()

        enabled = timeout_secs is not None and timeout_secs >= 0
        self.timer = threading.Timer(timeout_secs, timeout_handler) if enabled else None
        self.timed_out = False

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


__all__ = [s for s in dir() if not s.startswith('_') and s not in __no_export]
