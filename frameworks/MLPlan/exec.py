import logging
import math
import os
import json

from frameworks.shared.callee import call_run, result, output_subdir, utils

log = logging.getLogger(__name__)

def run(dataset, config):
    log.info("\n**** ML-Plans ****\n")

    is_classification = config.type == 'classification'
    
    # Mapping of benchmark metrics to Weka metrics
    metrics_mapping = dict(
        acc='ERRORRATE',
        auc='AUC',
        logloss='LOGLOSS',
        f1='F1',
        r2='R2',
        rmse='ROOT_MEAN_SQUARED_ERROR',
        mse='MEAN_SQUARED_ERROR',
        rmsle='ROOT_MEAN_SQUARED_LOGARITHM_ERROR',
        mae='MEAN_ABSOLUTE_ERROR'
    )
    metric = metrics_mapping[config.metric] if config.metric in metrics_mapping else None
    if metric is None:
	    raise ValueError('Performance metric {} is not supported.'.format(config.metric))
    
    train_file = dataset.train.path
    test_file = dataset.test.path
    # Weka requires target as the last attribute
    if dataset.target.index != len(dataset.predictors):
        train_file = reorder_dataset(dataset.train.path, target_src=dataset.target.index)
        test_file = reorder_dataset(dataset.test.path, target_src=dataset.target.index)

    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}
	
	backend = config.framework_params.get('_backend', 'weka')
	
    if backend == "weka":
        mem_limit = str(max(config.max_mem_size_mb-1024,2048))
    else:
        mem_limit = str(max((config.max_mem_size_mb-1024) / config.cores,2048))

    mode = backend
	if config.type == 'regression':
	    mode += '-regression'
	
    log.info("Running ML-Plan with backend %s in mode %s and a maximum time of %ss on %s cores with %sMB for the JVM, optimizing %s.", backend, mode, config.max_runtime_seconds, config.cores, config.max_mem_size_mb, metric)
    log.info("Environment: %s", os.environ)
	
	predictions_file = os.path.join(output_subdir('mlplan_out', config), 'predictions.csv')
	statistics_file = os.path.join(output_subdir('mlplan_out', config), 'statistics.json')

    cmd_root = "java -jar {here}/lib/mlplan/mlplan-cli*.jar -Xmx{mem_mb}M".format(here=dir_of(__file__),mem_mb=mem_limit)
    cmd_params = dict(
        f='"{}"'.format(train_file),
        p='"{}"'.format(test_file),
        t=config.max_runtime_seconds,
        ncpus=config.cores,
        l=metric,
		m=mode,
	    s=config.seed,   # weka accepts only int16 as seeds
	    ooab=predictions_file,
        **training_params
    )

    cmd = cmd_root + ' '.join(["-{} {}".format(k, v) for k, v in cmd_params.items()])

    with Timer() as training:
        run_cmd(cmd, _live_output_=True)

    return result(
	    output_file=config.output_predictions_file,
		predictions=predictions,
		truth=y_test,
		probabilities=probabilities,
		probabilities_labels=probabilities_labels,
		target_is_encoded=is_classification,
		models_count=models_count,
        training_duration=training.duration
    )

if __name__ == '__main__':
    call_run(run)