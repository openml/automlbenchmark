import datetime as dt
import logging

import automl


logger = logging.getLogger(automl.__name__)


class MillisFormatter(logging.Formatter):

    converter = dt.datetime.fromtimestamp

    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            t = ct.strftime(datefmt)
        else:
            t = ct.strftime("%Y-%m-%d %H:%M:%S")
        s = "%s.%03d" % (t, record.msecs)
        return s


def setup(log_file=None, root_file=None, root_level=logging.WARNING, app_level=None, console_level=None):
    root = logging.getLogger()
    root.setLevel(root_level)

    app_level = app_level if app_level else root_level
    console_level = console_level if console_level else app_level

    # create console handler
    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter('[%(levelname)s] [%(name)s] %(message)s'))
    logger.addHandler(console)

    file_formatter = MillisFormatter('[%(levelname)s] [%(name)s:%(asctime)s] %(message)s', datefmt='%H:%M:%S')

    if log_file:
        # create file handler
        file = logging.FileHandler(log_file, mode='w')
        file.setLevel(app_level)
        file.setFormatter(file_formatter)
        logger.addHandler(file)

    if root_file:
        file = logging.FileHandler(root_file, mode='w')
        file.setLevel(root_level)
        file.setFormatter(file_formatter)
        root.addHandler(file)

