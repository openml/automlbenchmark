""" Script to upload results from the benchmark to OpenML.
The benchmark run needs to be executed on OpenML datasets to be eligible for upload.
"""
import argparse
import contextlib
from contextlib import contextmanager
from datetime import datetime
import logging
import os
import pathlib
from typing import Optional

import openml
from openml import OpenMLRun

from amlb.defaults import default_dirs
from amlb.resources import config_load
from amlb.uploads import process_task_folder, missing_folds, _load_task_data

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def parse_args():
    description = "Script to upload results from the benchmark to OpenML."
    parser = argparse.ArgumentParser(description)
    parser.add_argument(
        '-i', '--input-directory', type=pathlib.Path, default=None, dest='input_directory',
        help="Directory that stores results from the runbenchmark.py invocation. "
             "By default use the most recent folder in the results "
             "folder as specified in the configuration."
    )
    parser.add_argument(
        '-a', '--api-key', dest='apikey', default=None,
        help="By default, the api key configured in your OpenML configuration file is"
             "used. Specify this parameter if you want to overwrite this value or"
             "you do not have an OpenML configuration file. A valid key is "
             "*required* to upload to the OpenML server."
             "This argument is ignored when uploading to the test server, as "
             "the default openml-python test server api key will be used instead."
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


def find_most_recent_result_folder() -> pathlib.Path:
    root_dir = pathlib.Path(__file__).parent
    config = config_load(root_dir / "resources" / "config.yaml")
    output_dir = pathlib.Path(config.output_dir or default_dirs.output_dir)

    def dirname_to_datetime(dirname: str) -> datetime:
        _, timestamp = dirname.rsplit('.', 1)
        return datetime.strptime(timestamp, "%Y%m%dT%H%M%S")

    run_directories = output_dir.glob("*.*.*.*")
    _, run_directory = max((dirname_to_datetime(str(d)), d) for d in run_directories)
    return run_directory


def resolve_input_directory(path: Optional[pathlib.Path]) -> pathlib.Path:
    path = path or find_most_recent_result_folder()
    path = path.expanduser().absolute()
    if not path.is_dir():
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


def server_for_task(task: pathlib.Path):
    metadata = _load_task_data(task)
    if metadata.test_server:
        server_connection = _connect_to_test_server()
    else:
        server_connection = contextlib.nullcontext()
    return server_connection


def upload_task(task_directory: pathlib.Path) -> Optional[OpenMLRun]:
    task_name = os.path.basename(task_directory)
    try:
        with server_for_task(task_directory):
            log.debug("Starting upload for '%s'." % task_name)
            run = process_task_folder(task_directory)
            log.info("%s result stored at %s/r/%d"
                     % (task_name, openml.config.server[:-11], run.id))
            return run
    except Exception as e:
        message = e.message if hasattr(e, "message") else e.args[0]
        log.warning("Task %s failed to upload: %s" % (task_name, message))
        if args.fail_fast:
            raise


def process_results(result_dir: pathlib.Path, mode: str = 'check'):
    prediction_directory = result_dir / "predictions"
    if not prediction_directory.exists():
        raise ValueError(f"result_dir '{result_dir!s}' has no predictions.")

    if args.task is None:
        tasks_to_process = [d for d in prediction_directory.iterdir() if d.is_dir()]
    elif (prediction_directory / args.task).is_dir():
        tasks_to_process = [args.task]
    else:
        raise ValueError(f"Task '{args.task}' not in '{prediction_directory}'.")

    for task_name in tasks_to_process:
        full_task_directory = prediction_directory / task_name

        folds = missing_folds(full_task_directory)
        if len(folds) > 0:
            log.info("%s has missing folds: %s" % (task_name, ', '.join(sorted(folds))))
            continue

        metadata = _load_task_data(full_task_directory)
        if "openml_task_id" not in metadata:
            log.info("%s has openml task metadata" % task_name)
            continue

        if mode == 'check':
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
        raise ValueError(
            "No valid OpenML API key configured, use the '--api-key' argument "
            "or follow instructions: https://openml.github.io/openml-python/master/usage.html#configuration"
        )
    input_directory = resolve_input_directory(args.input_directory)
    mode_verb = 'Uploading' if args.mode == 'upload' else 'Checking'
    log.info("%s results from '%s'." % (mode_verb, input_directory))

    process_results(input_directory, args.mode)
