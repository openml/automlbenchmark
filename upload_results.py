""" Script to upload results from the benchmark to OpenML.
The benchmark run needs to be executed on OpenML datasets to be eligible for upload.
"""
import argparse
import contextlib
from contextlib import contextmanager
from datetime import datetime
import logging
import os
from typing import Optional

import openml
from openml import OpenMLRun

from amlb.resources import config_load
from amlb.uploads import process_task_folder, missing_folds, _load_task_data

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def parse_args():
    description = "Script to upload results from the benchmark to OpenML."
    parser = argparse.ArgumentParser(description)
    parser.add_argument(
        '-i', type=str, default=None, dest='input_directory',
        help="Directory that stores results from the runbenchmark.py invocation. "
             "By default use the most recent folder in the results "
             "folder as specified in the configuration."
    )
    parser.add_argument(
        '-a', '--api-key', dest='apikey', default=None,
        help="OpenML API key to use for uploading results."
    )
    parser.add_argument(
        '-m', '--mode', dest='mode', default='check',
        help="Run mode (default=%(default)s)."
             "• check: only report whether results can be uploaded."
             "• upload: upload all complete results."
    )
    parser.add_argument(
        '-x', '--fail-fast', dest='fail_fast', action='store_true',
        help="Stop as soon as a task fails to upload due to an error during uploading."
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true', dest='verbose',
        help="Output progress to console."
    )
    parser.add_argument(
        '-t', '--task', type=str, dest='task', default=None,
        help="Only upload results for this specific task."
    )
    args = parser.parse_args()

    if args.mode not in ['check', 'upload']:
        raise ValueError(f"Invalid value for argument 'mode': '{args.mode}'.")

    return args


def find_most_recent_result_folder() -> str:
    root_dir = os.path.dirname(__file__)
    config = config_load(os.path.join(root_dir, "resources", "config.yaml"))

    def dir_to_datetime(dirname: str) -> datetime:
        timestamp = dirname.split('.')[-1]
        return datetime.strptime(timestamp, "%Y%m%dT%H%M%S")

    run_directories = [d for d in os.listdir(config.output_dir) if d.count('.') == 4]
    _, run_directory = max((dir_to_datetime(d), d) for d in run_directories)
    return os.path.join(config.output_dir, run_directory)


def resolve_input_directory(path: Optional[str]) -> str:
    if path is None:
        path = find_most_recent_result_folder()

    path = os.path.expanduser(path)
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        raise ValueError(f"{path} is not a directory.")
    return path


def configure_logging(verbose: bool):
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s',
        '%H:%M:%S',
    )
    log_level = logging.DEBUG if verbose else logging.INFO

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(log_level)

    log.addHandler(console)


@contextmanager
def _connect_to_test_server():
    openml.config.start_using_configuration_for_example()
    yield
    openml.config.stop_using_configuration_for_example()


def server_for_task(task: str):
    metadata = _load_task_data(task)
    if metadata.test_server:
        server_connection = _connect_to_test_server()
    else:
        server_connection = contextlib.nullcontext()
    return server_connection


def upload_task(task_directory: str) -> OpenMLRun:
    task_name = os.path.basename(task_directory)
    try:
        with server_for_task(task_directory):
            log.debug("Starting upload for '%s'." % task_name)
            run = process_task_folder(task_directory)
            log.info("%s result stored at %s/r/%d"
                     % (task_name, openml.config.server[:-11], run.id))
    except Exception as e:
        message = e.message if hasattr(e, "message") else e.args[0]
        log.warning("Task %s failed to upload: %s" % (task_name, message))
        if args.fail_fast:
            raise


def process_results(result_dir: str, mode: str = 'check'):
    prediction_directory = os.path.join(result_dir, 'predictions')

    if args.task is None:
        tasks_to_process = os.listdir(prediction_directory)
    elif os.path.isdir(os.path.join(prediction_directory, args.task)):
        tasks_to_process = [args.task]
    else:
        log.error(f"Task '%s' not in '%s'." % (args.task, prediction_directory))
        quit()

    for task_name in tasks_to_process:
        full_task_directory = os.path.join(prediction_directory, task_name)

        folds = missing_folds(full_task_directory)
        if len(folds) > 0:
            log.info("%s has missing folds: %s" % (task_name, ', '.join(sorted(folds))))
        elif mode == 'check':
            log.info("%s is ready for upload." % task_name)
        elif mode == 'upload':
            upload_task(full_task_directory)


def configure_apikey(key: Optional[str]) -> bool:
    if key:
        openml.config.apikey = key
    # API does not support easy checking of validity
    is_valid = True
    return openml.config.apikey is not None and is_valid


if __name__ == '__main__':
    args = parse_args()
    configure_logging(args.verbose)
    valid_key = configure_apikey(args.apikey)
    if not valid_key and args.mode == 'upload':
        log.error("No valid OpenML API key configured, use the '--api-key' argument "
                  "or follow instructions: https://openml.github.io/openml-python/master/usage.html#configuration")
        quit()

    input_directory = resolve_input_directory(args.input_directory)
    mode_verb = 'Uploading' if args.mode == 'upload' else 'Checking'
    log.info("%s results from '%s'." % (mode_verb, input_directory))

    process_results(input_directory, args.mode)
