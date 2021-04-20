import logging
import os
import re
from tempfile import TemporaryDirectory, mktemp
from typing import Union

import numpy as np

from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.resources import config as rconfig
from amlb.results import NoResultError, save_predictions
from amlb.utils import Namespace as ns, Timer, dir_of, run_cmd, json_dumps, json_load

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


def run_in_venv(caller_file, script_file: str, *args,
                input_data: Union[dict, ns], dataset: Dataset, config: TaskConfig,
                process_results=None,
                python_exec=None):

    here = dir_of(caller_file)
    venv_bin_path = os.path.join(here, 'venv', 'bin')
    if python_exec is None:  # use local virtual env by default
        python_exec = os.path.join(venv_bin_path, 'python -W ignore')
    script_path = os.path.join(here, script_file)
    cmd = f"{python_exec} {script_path}"

    input_data = ns.from_dict(input_data)
    with TemporaryDirectory() as tmpdir:

        def make_path(k, v, parents=None):
            if isinstance(v, np.ndarray):
                path = os.path.join(tmpdir, '.'.join(parents+[k, 'npy']))
                if vector_keys.match(k):
                    v = v.reshape(-1, 1)
                np.save(path, v, allow_pickle=True)
                return k, path
            return k, v

        ds = ns.walk(input_data, make_path)
        dataset.release()

        config.result_dir = tmpdir
        config.result_file = mktemp(dir=tmpdir)

        params = json_dumps(dict(dataset=ds, config=config), style='compact')
        with Timer() as proc_timer:
            output, err = run_cmd(cmd, *args,
                                  _input_str_=params,
                                  _live_output_=True,
                                  _error_level_=logging.DEBUG,
                                  _env_=dict(
                                      PATH=os.pathsep.join([
                                          venv_bin_path,
                                          os.environ['PATH']
                                      ]),
                                      PYTHONPATH=os.pathsep.join([
                                          rconfig().root_dir,
                                      ]),
                                      AMLB_PATH=os.path.join(rconfig().root_dir, "amlb")
                                    ),
                                  )

        res = ns(lambda: None)
        if os.path.exists(config.result_file):
            res = json_load(config.result_file, as_namespace=True)

        log.debug("Result from subprocess:\n%s", res)

        if not res:
            raise NoResultError(f"Process crashed:\n{err}")

        if res.error_message is not None:
            raise NoResultError(res.error_message)

        for name in ['predictions', 'truth', 'probabilities']:
            res[name] = np.load(res[name], allow_pickle=True) if res[name] is not None else None

        if callable(process_results):
            res = process_results(res)

        if res.output_file:
            save_predictions(dataset=dataset,
                             output_file=res.output_file,
                             predictions=res.predictions.reshape(-1) if res.predictions is not None else None,
                             truth=res.truth.reshape(-1) if res.truth is not None else None,
                             probabilities=res.probabilities,
                             probabilities_labels=res.probabilities_labels,
                             target_is_encoded=res.target_is_encoded)

        return dict(
            models_count=res.models_count if res.models_count is not None else 1,
            training_duration=res.training_duration if res.training_duration is not None else proc_timer.duration,
            predict_duration=res.predict_duration,
            **res.others.__dict__
        )
