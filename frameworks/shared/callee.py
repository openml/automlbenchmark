import json
import logging
import os
import re
import sys


class NS:

    @staticmethod
    def dict(ns, deep=True):
        dic = ns.__dict__
        if not deep:
            return dic
        for k, v in dic.items():
            if isinstance(v, NS):
                dic[k] = NS.dict(v)
        return dic

    @staticmethod
    def from_dict(dic, deep=True):
        ns = NS(dic)
        if not deep:
            return ns
        for k, v in ns.__dict__.items():
            if isinstance(v, dict):
                ns.__dict__[k] = NS.from_dict(v)
        return ns

    @staticmethod
    def walk(ns, fn, inplace=False):
        nns = ns if inplace else NS()
        for k, v in ns.__dict__.items():
            nk, nv = fn(k, v)
            if nk is not None:
                if v is nv and isinstance(v, NS):
                    nv = NS.walk(nv, fn, inplace)
                nns.__dict__[nk] = nv
        return nns

    def __init__(self, *args, **kwargs):
        self.__dict__.update(dict(*args, **kwargs))

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return repr(self.__dict__)


def setup_logger():
    console = logging.StreamHandler(sys.stdout)
    handlers = [console]
    logging.basicConfig(handlers=handlers)
    root = logging.getLogger()
    root.setLevel(logging.INFO)


def result(output_file=None,
           predictions=None, truth=None,
           probabilities=None, probabilities_labels=None,
           target_is_encoded=False,
           models_count=None,
           training_duration=None):
    return locals()


data_keys = re.compile("^(X|y|data)(_.+)?$")
setup_logger()


def call_run(run_fn):
    import numpy as np

    params = NS.from_dict(json.loads(sys.stdin.read()))

    def load_data(name, path):
        if isinstance(path, str) and data_keys.match(name):
            return name, np.load(path, allow_pickle=True)
        return name, path

    print(params.dataset)
    ds = NS.walk(params.dataset, load_data)

    config = params.config
    config.framework_params = NS.dict(config.framework_params)

    result = run_fn(ds, config)
    res = dict(result)

    for name in ['predictions', 'truth', 'probabilities']:
        arr = result[name]
        if arr is not None:
            res[name] = os.path.join(config.result_dir, '.'.join([name, 'npy']))
            np.save(res[name], arr, allow_pickle=True)

    print(config.result_token)
    print(json.dumps(res, separators=(',', ':')))
