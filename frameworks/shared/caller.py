import io
import logging
import os
import re
import uuid

import numpy as np

from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.results import save_predictions_to_file
from amlb.utils import Namespace as ns, TmpDir, dir_of, run_cmd, json_dumps, json_loads

log = logging.getLogger(__name__)

vector_keys = re.compile("^y(_.+)?$")


def call_process(cmd: str, send_data: ns, dataset: Dataset, config: TaskConfig):
    with TmpDir() as tmpdir:

        def make_path(k, v, parents=None):
            if isinstance(v, np.ndarray):
                path = os.path.join(tmpdir, '.'.join(parents+[k, 'npy']))
                if vector_keys.match(k):
                    v = v.reshape(-1, 1)
                np.save(path, v, allow_pickle=True)
                return k, path
            return k, v

        ds = ns.walk_apply(send_data, make_path)
        dataset.release()
        print(ds)

        config.result_token = str(uuid.uuid1())
        config.result_dir = tmpdir

        params = json_dumps(dict(dataset=ds, config=config), style='compact')
        output, err = run_cmd(cmd, _input_str_=params, _live_output_=False)

        out = io.StringIO(output)
        res = ns()
        for line in out:
            li = line.rstrip()
            if li == config.result_token:
                res = json_loads(out.readline(), as_namespace=True)
                break

        for name in ['predictions', 'truth', 'probabilities']:
            res[name] = np.load(res[name], allow_pickle=True) if res[name] is not None else None

        log.debug("Result from subprocess:\n%s", res)
        save_predictions_to_file(dataset=dataset,
                                 output_file=res.output_file,
                                 predictions=res.predictions.reshape(-1, 1),
                                 truth=res.truth.reshape(-1, 1),
                                 probabilities=res.probabilities,
                                 target_is_encoded=res.target_is_encoded)

