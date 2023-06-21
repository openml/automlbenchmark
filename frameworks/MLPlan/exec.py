import glob
import logging
import os
import json
import re
import tempfile

from frameworks.shared.callee import call_run, result, output_subdir
from frameworks.shared.utils import Timer, run_cmd

log = logging.getLogger(__name__)


def run(dataset, config):
    jar_file = glob.glob("{here}/lib/mlplan/mlplan-cli*.jar".format(here=os.path.dirname(__file__)))[0]
    version = re.match(r".*/mlplan-cli-(.*).jar", jar_file)[1]
    log.info(f"\n**** ML-Plan [v{version}] ****\n")

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

    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}
    backend = config.framework_params.get('_backend', 'weka')

    if backend == "weka":
        mem_limit = str(max(config.max_mem_size_mb-1024, 2048))
    else:
        mem_limit = str(max(round((config.max_mem_size_mb-1024) / config.cores), 2048))

    mode = backend
    if config.type == 'regression':
        mode += '-regression'

    log.info("Running ML-Plan with backend %s in mode %s and a maximum time of %ss on %s cores with %sMB for the JVM, optimizing %s.",
             backend, mode, config.max_runtime_seconds, config.cores, config.max_mem_size_mb, metric)
    log.info("Environment: %s", os.environ)

    mlplan_output_dir = output_subdir('mlplan_out', config)
    predictions_file = os.path.join(mlplan_output_dir, 'predictions.csv')
    statistics_file = os.path.join(mlplan_output_dir, 'statistics.json')

    cmd_root = f"java -jar -Xmx{mem_limit}M {jar_file}"

    with tempfile.TemporaryDirectory() as tmp_dir:
        cmd_params = dict(
            f='"{}"'.format(train_file),
            p='"{}"'.format(test_file),
            t=config.max_runtime_seconds,
            ncpus=config.cores,
            l=metric,
            m=mode,
            s=config.seed,   # weka accepts only int16 as seeds
            ooab=predictions_file,
            os=statistics_file,
            tmp=tmp_dir,
            **training_params
        )

        cmd = cmd_root + ''.join([" -{} {}".format(k, v) for k, v in cmd_params.items()])

        with Timer() as training:
            run_cmd(cmd, _live_output_=True)
        log.info(f"Finished fit in {training.duration}s.")

    with open(statistics_file, 'r') as f:
        stats = json.load(f)

    predictions = stats["predictions"]
    truth = stats["truth"]
    num_evals = stats["num_evaluations"]
    if "final_candidate_predict_time_ms" in stats:
        predict_time = stats["final_candidate_predict_time_ms"]
    else:
        predict_time = float("NaN")

    # only for classification tasks we have probabilities available, thus check whether the json contains the respective fields
    if "probabilities" in stats and "probabilities_labels" in stats:
        probabilities = stats["probabilities"]
        probabilities_labels = stats["probabilities_labels"]
    else:
        probabilities = None
        probabilities_labels = None

    if version == "0.2.3":
        target_encoded = is_classification
    else:
        target_encoded = False

    return result(
        output_file=config.output_predictions_file,
        predictions=predictions,
        truth=truth,
        probabilities=probabilities,
        probabilities_labels=probabilities_labels,
        target_is_encoded=target_encoded,
        models_count=num_evals,
        training_duration=training.duration,
        predict_duration=predict_time
    )


if __name__ == '__main__':
    call_run(run)
