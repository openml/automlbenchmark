import linecache
import json
import logging
import os
import re
import sys

try:
    # for backwards compatibility, imports the utility functions that used to be duplicated here
    from utils import Namespace as NS, Timer, touch
    import utils  # alias amlb utils
except ImportError:
    # callee was not imported from subprocess created by caller
    from amlb.utils import Namespace as NS, touch


def setup_logger():
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    handlers = [console]
    logging.basicConfig(handlers=handlers)
    root = logging.getLogger()
    root.setLevel(logging.INFO)


setup_logger()

log = logging.getLogger(__name__)


def result(output_file=None,
           predictions=None, truth=None,
           probabilities=None, probabilities_labels=None,
           target_is_encoded=False,
           error_message=None,
           models_count=None,
           training_duration=None,
           **others):
    return locals()


def output_subdir(name, config):
    subdir = os.path.join(config.output_dir, name, config.name, str(config.fold))
    touch(subdir, as_dir=True)
    return subdir


_extensions_ = {}


def get_extension(files, name=None, default=None):
    files = [files] if isinstance(files, str) else files

    extensions = []
    for file in files:
        if file in _extensions_:
            extensions.append(_extensions_.get(file, {}))
        elif os.path.isfile(file):
            try:
                with open(file) as f:
                    # linecache and compile are necessary only if we want to inspect code later
                    # otherwise the following statement is enough:
                    # exec(f.read(), customizations)
                    linecache.updatecache(f.name)
                    code = compile(f.read(), f.name, 'exec')
                    ext = {}
                    exec(code, ext)
                    _extensions_[file] = ext
                    extensions.append(ext)
            except Exception as e:
                log.warning("Could not load extension file %s: %s", file, str(e))
                _extensions_[file] = {}
        else:
            log.warning("No extensions available at %s", file)

    return extensions if name is None else next((ext[name] for ext in extensions if name in ext), default)


_data_keys_ = re.compile("^(X|y|data)(_.+)?$")


def call_run(run_fn):
    import numpy as np

    params = NS.from_dict(json.loads(sys.stdin.read()))

    def load_data(name, path, **ignored):
        if isinstance(path, str) and _data_keys_.match(name):
            return name, np.load(path, allow_pickle=True)
        return name, path

    print(params.dataset)
    ds = NS.walk(params.dataset, load_data)

    config = params.config
    config.framework_params = NS.dict(config.framework_params)

    try:
        result = run_fn(ds, config)
        res = dict(result)
        for name in ['predictions', 'truth', 'probabilities']:
            arr = result[name]
            if arr is not None:
                res[name] = os.path.join(config.result_dir, '.'.join([name, 'npy']))
                np.save(res[name], arr, allow_pickle=True)
    except Exception as e:
        log.exception(e)
        res = dict(
            error_message=str(e),
            models_count=0
        )

    print(config.result_token)
    print(json.dumps(res, separators=(',', ':')))
