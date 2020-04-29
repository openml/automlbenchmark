import io
import logging
import os
import re
from typing import Union
import uuid

import numpy as np

from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.resources import config as rconfig
from amlb.results import NoResultError, save_predictions_to_file
from amlb.utils import Namespace as ns, Timer, TmpDir, dir_of, run_cmd, json_dumps, json_loads

log = logging.getLogger(__name__)

vector_keys = re.compile("^y(_.+)?$")


def run_in_venv(caller_file, script_file: str, *args,
                input_data: Union[dict, ns], dataset: Dataset, config: TaskConfig,
                process_results=None,
                python_exec=None):

    here = dir_of(caller_file)
    if python_exec is None:  # use local virtual env by default
        python_exec = os.path.join(here, 'venv/bin/python -W ignore')
    script_path = os.path.join(here, script_file)
    cmd = f"{python_exec} {script_path}"

    input_data = ns.from_dict(input_data)
    with TmpDir() as tmpdir:

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

        config.result_token = str(uuid.uuid1())
        config.result_dir = tmpdir

        params = json_dumps(dict(dataset=ds, config=config), style='compact')
        with Timer() as proc_timer:
            output, err = run_cmd(cmd, *args,
                                  _input_str_=params,
                                  _live_output_=True,
                                  _env_=dict(
                                      PYTHONPATH=os.pathsep.join([
                                          rconfig().root_dir,
                                          os.path.join(rconfig().root_dir, "amlb"),
                                      ]))
                                  )

        out = io.StringIO(output)
        res = ns()
        for line in out:
            li = line.rstrip()
            if li == config.result_token:
                res = json_loads(out.readline(), as_namespace=True)
                break

        if res.error_message is not None:
            raise NoResultError(res.error_message)

        for name in ['predictions', 'truth', 'probabilities']:
            res[name] = np.load(res[name], allow_pickle=True) if res[name] is not None else None

        log.debug("Result from subprocess:\n%s", res)
        if callable(process_results):
            res = process_results(res)

        save_predictions_to_file(dataset=dataset,
                                 output_file=res.output_file,
                                 predictions=res.predictions.reshape(-1) if res.predictions is not None else None,
                                 truth=res.truth.reshape(-1) if res.truth is not None else None,
                                 probabilities=res.probabilities,
                                 target_is_encoded=res.target_is_encoded)

        return dict(
            models_count=res.models_count if res.models_count is not None else 1,
            training_duration=res.training_duration if res.training_duration is not None else proc_timer.duration,
            **res.others.__dict__
        )
