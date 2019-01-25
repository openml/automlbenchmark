"""
**logger** module just exposes a ``setup`` function to quickly configure the python logger.
"""
import datetime as dt
import io
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
        app_handler = logging.FileHandler(log_file, mode='a')
        app_handler.setLevel(app_level)
        app_handler.setFormatter(file_formatter)
        app_logger.addHandler(app_handler)
        frameworks_logger.addHandler(app_handler)

    if root_file:
        root_handler = logging.FileHandler(root_file, mode='a')
        root_handler.setLevel(root_level)
        root_handler.setFormatter(file_formatter)
        root.addHandler(root_handler)

    if print_to_log:
        import builtins
        nl = '\n'
        print_logger = logging.getLogger(app_logger.name + '.print')
        buffer = dict(out=None, err=None)

        ori_print = builtins.print
        def new_print(self, *args, sep=' ', end=nl, file=None):
            if file not in [None, sys.stdout, sys.stderr]:
                return ori_print(self, *args, sep=sep, end=end, file=file)

            nonlocal buffer
            buf = buffer['err'] if file is sys.stderr else buffer['out']
            buf = buf if buf is not None else io.StringIO()
            buf.write(sep.join([self, *args]))  # end added by logger
            if end == nl:
                with buf:
                    if file is sys.stderr:
                        print_logger.error(buf.getvalue())
                        buffer['err'] = None
                        # ori_print(traceback.format_stack())
                    else:
                        print_logger.info(buf.getvalue())
                        buffer['out'] = None

        builtins.print = new_print
