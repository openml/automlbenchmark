import logging
import os

from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.datautils import impute
from amlb.utils import Namespace as ns, dir_of

from .caller import call_process

log = logging.getLogger(__name__)
PYTHON = os.path.join(dir_of(__file__), 'venv/bin/python3 -W ignore')


def run(dataset: Dataset, config: TaskConfig):
    X_train_enc, X_test_enc = impute(dataset.train.X_enc, dataset.test.X_enc)
    data = ns(
        train=ns(
            X_enc=X_train_enc,
            y=dataset.train.y
        ),
        test=ns(
            X_enc=X_test_enc,
            y=dataset.test.y
        )
    )

    return call_process("{python} {here}/exec_proc.py".format(python=PYTHON, here=dir_of(__file__)),
                        send_data=data,
                        dataset=dataset,
                        config=config)

