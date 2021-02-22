import logging
import math
import os

from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.datautils import reorder_dataset
from amlb.results import NoResultError, save_predictions
from amlb.utils import dir_of, path_from_split, run_cmd, split_path, Timer

from frameworks.shared.callee import save_metadata

import logging
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import tempfile as tmp
import warnings
import optuna.integration.lightgbm as lgb
from frameworks.shared.callee import call_run, result, output_subdir, utils
from amlb.utils import Timer
import pandas as pd
import time, threading, _thread
from amlb.benchmark import TaskConfig
from amlb.data import Dataset

log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    log.info("\n**** Optuna LGBM ****\n")
    time_budget = config.max_runtime_seconds
    n_jobs = config.framework_params.get('_n_jobs', config.cores)

    print("Running Optuna LGBM with {} number of cores".format(config.cores))

    save_metadata(config)

    is_classification = config.type == 'classification'
    enc = config.framework_params.get('_enc', False)
    if enc:
        import numpy as np
        X_train = dataset.train.X_enc
        y_train = dataset.train.y_enc
        X_test = dataset.test.X_enc
        y_test = dataset.test.y_enc
        train = label = None
    else:
        column_names, _ = zip(*dataset.columns)
        column_types = dict(dataset.columns)
        train = pd.DataFrame(dataset.train.data, columns=column_names).astype(
            column_types, copy=False)        
        label = dataset.target.name
        X_train = train.drop(columns=label)
        y_train = train[label]
        test = pd.DataFrame(dataset.test.data, columns=column_names).astype(
            column_types, copy=False)
        X_test = test.drop(columns=label)
        y_test = test[label]
        from sklearn.model_selection import train_test_split
        print('dataset preparing')
        train_x, val_x, train_y, val_y = train_test_split(
            X_train, y_train, test_size=0.1)
        dtrain = lgb.Dataset(train_x, label=train_y)
        dval = lgb.Dataset(val_x, label=val_y)
        print('dataset prepared')
    params = {
        "objective": "regression",
        "metric": "regression",
        "verbosity": -1,
        # "boosting_type": "gbdt",
    }

    training_params = {k: v for k, v in config.framework_params.items()
     if not k.startswith('_')}

    with Timer() as training:
        model = lgb.train(
            params, dtrain, valid_sets=[dtrain, dval], verbose_eval=10000)        
    
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test) if is_classification else None
    labels = model.classes_ if is_classification else None
    return result(  
                    output_file=config.output_predictions_file,
                    probabilities=probabilities,
                    predictions=predictions,
                    # target_is_encoded=is_classification and enc,
                    truth=y_test,
                    models_count=0,
                    training_duration=training.duration,
                    probabilities_labels=labels,
                )

if __name__ == '__main__':
    call_run(run)