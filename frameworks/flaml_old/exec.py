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
from flaml import AutoML
from frameworks.shared.callee import call_run, result, output_subdir, utils
from amlb.utils import Timer
import pandas as pd
import time, threading, _thread
from amlb.benchmark import TaskConfig
from amlb.data import Dataset

log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    log.info("\n**** FLAM L0.1.3 ****\n")
    time_budget = config.max_runtime_seconds
    n_jobs = config.framework_params.get('_n_jobs', config.cores)

    print("Running FLAML with {} number of cores".format(config.cores))

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
        X_train = y_train = None
        test = pd.DataFrame(dataset.test.data, columns=column_names).astype(
            column_types, copy=False)
        X_test = test.drop(columns=label)
        y_test = test[label]

    aml = AutoML()

    # Mapping of benchmark metrics to flaml metrics
    metrics_mapping = dict(
        acc='accuracy',
        auc='roc_auc',
        f1='f1',
        logloss='log_loss',
        mae='mae',
        mse='mse',
        rmse='rmse',
        r2='r2',
    )

    metric = metrics_mapping[config.metric] if config.metric in metrics_mapping else None
    if metric is None:
        raise ValueError("Performance metric {} not supported.".format(config.metric))

    perf_metric = metrics_mapping[
        config.metric] if config.metric in metrics_mapping else 'auto'
    if perf_metric is None:
        log.warning("Performance metric %s not supported.", config.metric)

    training_params = {k: v for k, v in config.framework_params.items()
     if not k.startswith('_')}

    with Timer() as training:
        aml.fit(X_train, y_train, train, label, perf_metric, config.type, 
            n_jobs=n_jobs, 
            log_file_name=config.output_predictions_file.replace('.csv','.log'),
            time_budget=time_budget, **training_params)
    
    predictions = aml.predict(X_test)
    probabilities = aml.predict_proba(X_test) if is_classification else None
    labels = aml.classes_ if is_classification else None
    return result(  
                    output_file=config.output_predictions_file,
                    probabilities=probabilities,
                    predictions=predictions,
                    # target_is_encoded=is_classification and enc,
                    truth=y_test,
                    models_count=len(aml.config_history),
                    training_duration=training.duration,
                    probabilities_labels=labels,
                )

if __name__ == '__main__':
    call_run(run)