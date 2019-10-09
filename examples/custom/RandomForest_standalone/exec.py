import io
import logging
import os
import uuid

from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.datautils import write_csv, read_csv
from amlb.results import save_predictions_to_file
from amlb.utils import Namespace as ns, TmpDir, dir_of, run_cmd, json_dumps, json_loads

log = logging.getLogger(__name__)


PYTHON = os.path.join(dir_of(__file__), 'venv/bin/python3 -W ignore')
# PYTHON = 'python3 -W ignore'


def run(dataset: Dataset, config: TaskConfig):
    with TmpDir() as tmpdir:
        ds = ns(
            train=ns(
                X_enc=os.path.join(tmpdir, 'train.X_enc'),
                y=os.path.join(tmpdir, 'train.y')
            ),
            test=ns(
                X_enc=os.path.join(tmpdir, 'test.X_enc'),
                y=os.path.join(tmpdir, 'test.y')
            )
        )
        write_csv(dataset.train.X_enc, ds.train.X_enc),
        write_csv(dataset.train.y.reshape(-1, 1), ds.train.y),
        write_csv(dataset.test.X_enc, ds.test.X_enc),
        write_csv(dataset.test.y.reshape(-1, 1), ds.test.y),
        dataset.release()
        config.result_token = str(uuid.uuid1())
        config.result_dir = tmpdir
        params = json_dumps(dict(dataset=ds, config=config), style='compact')
        output, err = run_cmd('{python} {here}/exec_proc.py'.format(python=PYTHON, here=dir_of(__file__)), _input_str_=params)
        out = io.StringIO(output)
        res = ns()
        for line in out:
            li = line.rstrip()
            if li == config.result_token:
                res = json_loads(out.readline(), as_namespace=True)
                break

        def load_data(path):
            return read_csv(path, as_data_frame=False, header=False)

        log.debug("Result from subprocess:\n%s", res)
        save_predictions_to_file(dataset=dataset,
                                 output_file=res.output_file,
                                 probabilities=load_data(res.probabilities) if res.probabilities is not None else None,
                                 predictions=load_data(res.predictions).squeeze(),
                                 truth=load_data(res.truth).squeeze(),
                                 target_is_encoded=res.target_is_encoded)

