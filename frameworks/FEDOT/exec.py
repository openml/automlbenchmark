import logging
import os
from pathlib import Path
from typing import Union

import pandas as pd

from fedot.api.main import Fedot

from frameworks.shared.callee import call_run, result, output_subdir, measure_inference_times
from frameworks.shared.utils import Timer

log = logging.getLogger(__name__)


def run(dataset, config):
    log.info("\n**** FEDOT ****\n")

    is_classification = config.type == 'classification'
    scoring_metric = get_fedot_metrics(config)

    training_params = {"preset": "best_quality", "n_jobs": config.cores}
    training_params.update({k: v for k, v in config.framework_params.items() if not k.startswith('_')})
    n_jobs = training_params["n_jobs"]

    log.info(f"Running FEDOT with a maximum time of {config.max_runtime_seconds}s on {n_jobs} cores, \
             optimizing {scoring_metric}")
    runtime_min = config.max_runtime_seconds / 60
    fedot = Fedot(problem=config.type, timeout=runtime_min, metric=scoring_metric, seed=config.seed,
                  max_pipeline_fit_time=runtime_min / 10, **training_params)


    with Timer() as training:
        fedot.fit(features=dataset.train.X, target=dataset.train.y)
    
    def infer(data: Union[str, pd.DataFrame]):
        test_data = pd.read_parquet(data) if isinstance(data, str) else data
        predict_fn = fedot.predict_proba if is_classification else fedot.predict
        return predict_fn(test_data)
    
    inference_times = {}
    if config.measure_inference_time:
        inference_times["file"] = measure_inference_times(infer, dataset.inference_subsample_files)
        inference_times["df"] = measure_inference_times(
            infer,
            [(1, dataset.test.X.sample(1, random_state=i)) for i in range(100)],
        )
        log.info(f"Finished inference time measurements.")
    
    log.info('Predicting on the test set.')
    with Timer() as predict:
        predictions = fedot.predict(features=dataset.test.X)
        probabilities = None
        if is_classification:
            probabilities = fedot.predict_proba(features=dataset.test.X, probs_for_all_classes=True)

    save_artifacts(fedot, config)

    return result(output_file=config.output_predictions_file,
                  predictions=predictions,
                  truth=dataset.test.y,
                  probabilities=probabilities,
                  target_is_encoded=False,
                  models_count=fedot.current_pipeline.length,
                  training_duration=training.duration,
                  predict_duration=predict.duration,
                  inference_times=inference_times,)


def get_fedot_metrics(config):
    metrics_mapping = dict(
        acc='accuracy',
        auc='roc_auc',
        f1='f1',
        logloss='neg_log_loss',
        mae='mae',
        mse='mse',
        msle='msle',
        r2='r2',
        rmse='rmse',
    )
    scoring_metric = metrics_mapping.get(config.metric, None)

    if scoring_metric is None:
        log.warning(f"Performance metric {config.metric} not supported.")

    return scoring_metric


def save_artifacts(automl, config):

    artifacts = config.framework_params.get('_save_artifacts', [])
    if 'models' in artifacts:
        try:
            models_dir = output_subdir('models', config)
            models_file = os.path.join(models_dir, 'model.json')
            automl.current_pipeline.save(models_file)
        except Exception as e:
            log.info(f"Error when saving 'models': {e}.", exc_info=True)

    if 'info' in artifacts:
        try:
            info_dir = output_subdir("info", config)
            if automl.history:
                automl.history.save(os.path.join(info_dir, 'history.json'))
            else:
                log.info(f"There is no optimization history info to save.")
        except Exception as e:
            log.info(f"Error when saving info about optimisation history: {e}.", exc_info=True)

    if 'leaderboard' in artifacts:
        try:
            leaderboard_dir = output_subdir("leaderboard", config)
            if automl.history:
                lb = automl.history.get_leaderboard()
                Path(os.path.join(leaderboard_dir, "leaderboard.csv")).write_text(lb)
        except Exception as e:
            log.info(f"Error when saving 'leaderboard': {e}.", exc_info=True)


if __name__ == '__main__':
    call_run(run)
