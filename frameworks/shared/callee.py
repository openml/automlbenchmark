import os
import re
import sys

import numpy as np

from .core import Namespace as ns, json_dumps, json_loads


def result(output_file=None,
           predictions=None, truth=None,
           probabilities=None, probabilities_labels=None,
           target_is_encoded=False,
           models_count=None,
           training_duration=None):
    return locals()


data_keys = re.compile("^(X|y|data)(_.+)?$")


def call_run(run_fn):
    params = json_loads(sys.stdin.read(), as_namespace=True)

    def load_data(name, path, **_):
        if isinstance(path, str) and data_keys.match(name):
            return name, np.load(path, allow_pickle=True)
        return name, path

    ds = ns.walk_apply(params.dataset, load_data)

    config = params.config
    config.framework_params = ns.dict(config.framework_params)

    result = run_fn(ds, config)
    res = dict(result)

    for name in ['predictions', 'truth', 'probabilities']:
        arr = result[name]
        if arr is not None:
            res[name] = os.path.join(config.result_dir, '.'.join([name, 'npy']))
            np.save(res[name], arr, allow_pickle=True)

    print(config.result_token)
    print(json_dumps(res, style='compact'))
