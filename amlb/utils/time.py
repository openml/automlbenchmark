import datetime as dt
import logging
import math
import threading
import time

from .core import identity, threadsafe_generator

log = logging.getLogger(__name__)


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


def countdown(timeout_secs, on_timeout=None, message=None, frequency=1, log_level=logging.INFO):
    timeout_epoch = time.time() + timeout_secs
    remaining = timeout_secs
    while remaining > 0:
        mins, secs = divmod(remaining, 60)
        hours, mins = divmod(mins, 60)
        if message:
            log.log(log_level, "in %02d:%02d:%02d : %s", hours, mins, secs, message)
        else:
            log.log(log_level, "countdown: %02d:%02d:%02d", hours, mins, secs)
        next_sleep = min(frequency, remaining)
        time.sleep(next_sleep)
        remaining = math.ceil(timeout_epoch - time.time())
    if on_timeout:
        on_timeout()


@threadsafe_generator
def retry(start=0, fn=identity, max_retries=-1):
    delay = start
    retries = 1
    while True:
        if 0 <= max_retries < retries:
            return
        yield delay
        retries = retries+1
        delay = fn(delay)


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

