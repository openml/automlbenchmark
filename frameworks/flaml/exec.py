import logging
import warnings
import pandas as pd

from amlb.utils import Timer

from flaml import AutoML, __version__

from frameworks.shared.callee import call_run, result, save_metadata

log = logging.getLogger(__name__)


def run(dataset, config):
    log.info(f"\n**** FLAML [v{__version__}] ****\n")
    save_metadata(config, version=__version__)
    time_budget = config.max_runtime_seconds
    n_jobs = config.framework_params.get('_n_jobs', config.cores)

    print("Running FLAML with {} number of cores".format(config.cores))

    is_classification = config.type == 'classification'
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
                    truth=y_test,
                    models_count=len(aml.config_history),
                    training_duration=training.duration,
                    probabilities_labels=labels,
                )

if __name__ == '__main__':
    call_run(run)