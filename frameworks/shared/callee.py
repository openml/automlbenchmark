import logging
import os
import re
import signal
import sys

from .utils import InterruptTimeout, Namespace as ns, json_dump, json_loads, kill_proc_tree, touch
from .utils import deserialize_data, serialize_data

log = logging.getLogger(__name__)


class FrameworkError(Exception):
    pass


def result(output_file=None,
           predictions=None, truth=None,
           probabilities=None, probabilities_labels=None,
           optional_columns=None,
           target_is_encoded=False,
           error_message=None,
           models_count=None,
           training_duration=None,
           predict_duration=None,
           **others):
    return locals()


def output_subdir(name, config):
    subdir = os.path.join(config.output_dir, name, config.name, str(config.fold))
    touch(subdir, as_dir=True)
    return subdir


def save_metadata(config, **kwargs):
    obj = dict(config.__dict__)
    obj.update(kwargs)
    json_dump(obj, config.output_metadata_file, style='pretty')


data_keys = re.compile("^(X|y|data)(_.+)?$")


def call_run(run_fn):
    # log.info(os.environ)
    params = ns.from_dict(json_loads(sys.stdin.read()))
    ser_config = params.options['serialization']

    def load_data(name, path, **_):
        if isinstance(path, str) and data_keys.match(name):
            return name, deserialize_data(path, config=ser_config)
        return name, path

    log.debug("Params read from main process:\n%s", params)
    ds = ns.walk(params.dataset, load_data)

    config = params.config
    config.framework_params = ns.dict(config.framework_params)

    try:
        with InterruptTimeout(config.job_timeout_seconds,
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
            for name in ['predictions', 'truth', 'probabilities', 'optional_columns']:
                arr = result[name]
                if arr is not None:
                    path = os.path.join(config.result_dir, '.'.join([name, 'data']))
                    res[name] = serialize_data(arr, path, config=ser_config)
    except BaseException as e:
        log.exception(e)
        res = dict(
            error_message=str(e),
            models_count=0
        )
    finally:
        # ensure there's no subprocess left
        kill_proc_tree(include_parent=False, timeout=5)

    json_dump(res, config.result_file, style='compact')
