import gc
import logging
import os
import pathlib
import re
from tempfile import TemporaryDirectory, mktemp
from typing import List, Optional, Union

import numpy as np

from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.resources import config as rconfig
from amlb.results import NoResultError, save_predictions
from amlb.utils import json_dump, Namespace

from .utils import Namespace as ns, Timer, dir_of, run_cmd, json_dumps, json_load, profile
from .utils import is_serializable_data, deserialize_data, serialize_data

log = logging.getLogger(__name__)

vector_keys = re.compile("^y(_.+)?$")


def run_cmd_in_venv(caller_file, cmd, *args, **kwargs):
    params = ns(
        python_exec='python'
    )
    for k, v in params:
        kk = '_'+k+'_'
        if kk in kwargs:
            params[k] = kwargs[kk]
            del kwargs[kk]

    here = dir_of(caller_file)
    venv_bin_path = os.path.join(here, 'venv', 'bin')
    if os.path.isdir(venv_bin_path):
        py = os.path.join(venv_bin_path, 'python -W ignore')
        pip = os.path.join(venv_bin_path, 'python -m pip')
    else:
        py = f"{params.python_exec} -W ignore"
        pip = f"{params.python_exec} -m pip"

    cmd = cmd.format(py=py, pip=pip)
    return run_cmd(cmd, *args, **kwargs)


def as_vec(data):
    return data.reshape(-1) if isinstance(data, np.ndarray) else data


def as_col(data):
    return data.reshape(-1, 1) if isinstance(data, np.ndarray) else data


def venv_bin(fmwk_dir):
    return os.path.join(fmwk_dir, 'venv', 'bin')


def venv_python_exec(fmwk_dir):
    return os.path.join(venv_bin(fmwk_dir), 'python -W ignore')


@profile(logger=log)
def _make_input_dataset(input_data, dataset, tmpdir, serialization: Optional[ns] = None):
    input_data = ns.from_dict(input_data)

    def make_path(k, v, parents=None):
        if is_serializable_data(v):
            path = os.path.join(tmpdir, '.'.join(parents+[k, 'data']))
            if vector_keys.match(k):
                v = as_col(v)
            path = serialize_data(v, path, config=serialization)
            return k, path
        return k, v

    ds = ns.walk(input_data, make_path)
    dataset.release()
    gc.collect()
    return ds


def run_in_venv(caller_file, script_file: str, *args,
                input_data: Union[dict, ns], dataset: Dataset, config: TaskConfig,
                options: Union[None, dict, ns] = None,
                process_results=None,
                python_exec=None,
                retained_env_vars: Optional[List[str]] = ['TMP', 'TEMP', 'TMPDIR']):
    here = dir_of(caller_file)
    if python_exec is None:  # use local virtual env by default
        python_exec = venv_python_exec(here)
    script_path = os.path.join(here, script_file)
    cmd = f"{python_exec} {script_path}"

    options = ns.from_dict(options) if options else ns()
    ser_config = options['serialization']
    env = options['env'] or ns()

    # Add any env variables specified if they are defined in the environment
    if retained_env_vars:
        for env_var in retained_env_vars:
            env_val = os.environ.get(env_var)
            if env_val is not None:
                env[env_var] = env_val

    with TemporaryDirectory(prefix='amlb_', suffix='_xproc') as tmpdir:

        ds = _make_input_dataset(input_data, dataset, tmpdir, serialization=ser_config)

        config.result_dir = tmpdir
        config.result_file = mktemp(dir=tmpdir)

        params = json_dumps(dict(dataset=ds, config=config, options=options), style='compact')
        log.debug("Params passed to subprocess:\n%s", params)
        cmon = rconfig().monitoring
        monitor = (dict(interval_seconds=cmon.interval_seconds,
                        verbosity=cmon.verbosity)
                   if 'sub_proc_memory' in cmon.statistics
                   else None)
        env = dict(
            PATH=os.pathsep.join([
                venv_bin(here),
                os.environ['PATH']
            ]),
            PYTHONPATH=os.pathsep.join([
                rconfig().root_dir,
            ]),
            AMLB_PATH=os.path.join(rconfig().root_dir),
            AMLB_LOG_TRACE=str(logging.TRACE if hasattr(logging, 'TRACE') else ''),
            **{k: str(v) for k, v in env}
        )

        with Timer() as proc_timer:
            output, err = run_cmd(cmd, *args,
                                  _input_str_=params,
                                  _live_output_=True,
                                  _error_level_=logging.DEBUG,
                                  _env_=env,
                                  _monitor_=monitor
                                  )

        res = ns(lambda: None)
        if os.path.exists(config.result_file):
            res = json_load(config.result_file, as_namespace=True)

        log.debug("Result from subprocess:\n%s", res)

        if not res:
            raise NoResultError(f"Process crashed:\n{err}")

        if res.error_message is not None:
            raise NoResultError(res.error_message)

        for name in ['predictions', 'truth', 'probabilities', 'optional_columns']:
            res[name] = deserialize_data(res[name], config=ser_config) if res[name] is not None else None

        inference_filepath = Namespace.dict(res.others).get("inference_times")
        if inference_filepath:
            inference_times = json_load(inference_filepath)
            inference_filepath = pathlib.Path(res.output_file).parent / "inference.json"
            json_dump(inference_times, inference_filepath)
            res["others"]["inference_times"] = inference_times

        if callable(process_results):
            res = process_results(res)

        if res.output_file:
            save_predictions(dataset=dataset,
                             output_file=res.output_file,
                             predictions=as_vec(res.predictions),
                             truth=(as_vec(res.truth) if res.truth is not None
                                    else dataset.test.y_enc if res.target_is_encoded
                                    else dataset.test.y),
                             probabilities=res.probabilities,
                             probabilities_labels=res.probabilities_labels,
                             optional_columns=res.optional_columns,
                             target_is_encoded=res.target_is_encoded)

        return dict(
            models_count=res.models_count if res.models_count is not None else 1,
            training_duration=res.training_duration if res.training_duration is not None else proc_timer.duration,
            predict_duration=res.predict_duration,
            **res.others.__dict__
        )
