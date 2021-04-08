import importlib.util
import logging
import os
import re
import signal
import sys


class FrameworkError(Exception):
    pass


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


amlb_path = os.environ.get("AMLB_PATH")
if amlb_path:
    utils = load_module("amlb.utils", os.path.join(amlb_path, "utils", "__init__.py"))
else:
    import amlb.utils as utils


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
           predict_duration=None,
           **others):
    return locals()


def output_subdir(name, config):
    subdir = os.path.join(config.output_dir, name, config.name, str(config.fold))
    utils.touch(subdir, as_dir=True)
    return subdir


def save_metadata(config, **kwargs):
    obj = dict(config.__dict__)
    obj.update(kwargs)
    utils.json_dump(obj, config.output_metadata_file, style='pretty')


data_keys = re.compile("^(X|y|data)(_.+)?$")


def call_run(run_fn):
    import numpy as np

    params = utils.Namespace.from_dict(utils.json_loads(sys.stdin.read()))

    def load_data(name, path, **ignored):
        if isinstance(path, str) and data_keys.match(name):
            return name, np.load(path, allow_pickle=True)
        return name, path

    log.info("Params passed to subprocess:\n%s", params)
    ds = utils.Namespace.walk(params.dataset, load_data)

    config = params.config
    config.framework_params = utils.Namespace.dict(config.framework_params)

    try:
        with utils.InterruptTimeout(config.job_timeout_seconds,
                                    interruptions=[
                                        dict(sig=TimeoutError),
                                        dict(sig=signal.SIGTERM),
                                        dict(sig=signal.SIGQUIT),
                                        dict(sig=signal.SIGKILL),
                                        dict(interrupt='process', sig=signal.SIGKILL)
                                    ],
                                    wait_retry_secs=10):
            result = run_fn(ds, config)
            res = dict(result)
            for name in ['predictions', 'truth', 'probabilities']:
                arr = result[name]
                if arr is not None:
                    res[name] = os.path.join(config.result_dir, '.'.join([name, 'npy']))
                    np.save(res[name], arr, allow_pickle=True)
    except BaseException as e:
        log.exception(e)
        res = dict(
            error_message=str(e),
            models_count=0
        )
    finally:
        # ensure there's no subprocess left
        utils.kill_proc_tree(include_parent=False, timeout=5)

    utils.json_dump(res, config.result_file, style='compact')
