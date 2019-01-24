"""
**logger** module just exposes a ``setup`` function to quickly configure the python logger.
"""
import builtins
import datetime as dt
import logging
import sys


app_logger = logging.getLogger('automl')
frameworks_logger = logging.getLogger('frameworks')

logging.TRACE = logging.TRACE if hasattr(logging, 'TRACE') else 5


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


def setup(log_file=None, root_file=None, root_level=logging.WARNING, app_level=None, console_level=None, print_to_log=False):
    """
    configures the Python logger.
    :param log_file:
    :param root_file:
    :param root_level:
    :param app_level:
    :param console_level:
    :return:
    """
    logging.captureWarnings(True)
    # warnings = logging.getLogger('py.warnings')

    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")

    root = logging.getLogger()
    root.setLevel(root_level)

    app_level = app_level if app_level else root_level
    console_level = console_level if console_level else app_level

    # create console handler
    console = logging.StreamHandler()
    console.setLevel(console_level)
    # console.setFormatter(logging.Formatter('[%(levelname)s] [%(name)s] %(message)s'))
    app_logger.addHandler(console)
    frameworks_logger.addHandler(console)

    file_formatter = MillisFormatter('[%(levelname)s] [%(name)s:%(asctime)s] %(message)s', datefmt='%H:%M:%S')

    if log_file:
        # create file handler
        file = logging.FileHandler(log_file, mode='a')
        file.setLevel(app_level)
        file.setFormatter(file_formatter)
        app_logger.addHandler(file)
        frameworks_logger.addHandler(file)

    if root_file:
        file = logging.FileHandler(root_file, mode='a')
        file.setLevel(root_level)
        file.setFormatter(file_formatter)
        root.addHandler(file)

    if print_to_log:
        nl = '\n'
        print_logger = logging.getLogger(app_logger.name + '.print')
        buffer = []

        def new_print(self, *args, sep=' ', end=nl, file=None):
            nonlocal buffer
            msg = sep.join([self, *args]) #+ end
            buffer.append(msg)
            if end == nl:
                print_logger.info(''.join(buffer))
                buffer = []

        builtins.print = new_print