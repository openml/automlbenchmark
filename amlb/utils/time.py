import datetime as dt
import logging
import threading
import time

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


