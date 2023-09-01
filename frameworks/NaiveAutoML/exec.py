import logging
import os
import pickle
import re
import subprocess
import sys
import tempfile as tmp
from pathlib import Path
from typing import Union

import pandas as pd

if sys.platform == 'darwin':
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from frameworks.shared.callee import call_run, result, output_subdir, \
    measure_inference_times
from frameworks.shared.utils import Timer, touch

from naiveautoml import NaiveAutoML

log = logging.getLogger(__name__)


def run(dataset, config):
    pip_list = subprocess.run("python -m pip list".split(), capture_output=True)
    match = re.search(r"naiveautoml\s+([^\n]+)", pip_list.stdout.decode(), flags=re.IGNORECASE)
    version, = match.groups()
    log.info("\n**** NaiveAutoML [v%s] ****", version)

    metrics_mapping = dict(
        acc='accuracy',
        balacc='balanced_accuracy',
        auc='roc_auc',
        logloss='neg_log_loss',
        mae='neg_mean_absolute_error',
        r2='r2',
        rmse='neg_mean_squared_error',
    )
    scoring_metric = metrics_mapping.get(config.metric)
    if scoring_metric is None:
        raise ValueError(f"Performance metric {config.metric} not supported.")

    kwargs = dict(
        scoring=scoring_metric,
        num_cpus=config.cores,
    )
    # NAML wasn't really designed to run for long time constraints, so we
    # make it easy to run NAML with its default configuration for time/iterations.
    if not config.framework_params.get("_use_default_time_and_iterations", False):
        kwargs["timeout"] = config.max_runtime_seconds
        # NAML stops at its first met criterion: iterations or time.
        # To ensure time is the first criterion, set max_hpo_iterations very high
        kwargs["max_hpo_iterations"] = 1e10
        # NAML has a static per-pipeline evaluation time of 10 seconds,
        # which is not accommodation for larger datasets.
        kwargs["execution_timeout"] = max(config.max_runtime_seconds // 20, 10)
    else:
        log.info("`_use_default_time_and_iterations` is set, ignoring time constraint.")

    kwargs |= {k: v for k, v in config.framework_params.items() if not k.startswith("_")}
    automl = NaiveAutoML(**kwargs)

    with Timer() as training:
        automl.fit(dataset.train.X, dataset.train.y)
    log.info(f"Finished fit in {training.duration}s.")

    is_classification = (config.type == 'classification')

    def infer(data: Union[str, pd.DataFrame]):
        test_data = pd.read_parquet(data) if isinstance(data, str) else data
        predict_fn = automl.predict_proba if is_classification else automl.predict
        return predict_fn(test_data)

    inference_times = {}
    if config.measure_inference_time:
        inference_times["file"] = measure_inference_times(infer, dataset.inference_subsample_files)
        inference_times["df"] = measure_inference_times(
            infer,
            [(1, dataset.test.X.sample(1, random_state=i)) for i in range(100)],
        )
        log.info(f"Finished inference time measurements.")

    with Timer() as predict:
        predictions = automl.predict(dataset.test.X)
        probabilities = automl.predict_proba(dataset.test.X) if is_classification else None
    log.info(f"Finished predict in {predict.duration}s.")

    save_artifacts(automl, config)

    return result(
        output_file=config.output_predictions_file,
        predictions=predictions,
        probabilities=probabilities,
        truth=dataset.test.y,
        # models_count=len(gama_automl._final_pop),
        training_duration=training.duration,
        predict_duration=predict.duration,
        inference_times=inference_times,
        target_is_encoded=is_classification,
    )


def save_artifacts(naive_automl, config):
    artifacts = config.framework_params.get('_save_artifacts', ['history'])
    try:
        artifacts_dir = Path(output_subdir("artifacts", config))
        if 'history' in artifacts:
            naive_automl.history.to_csv(artifacts_dir / "history.csv", index=False)

        if 'model' in artifacts:
            (artifacts_dir / "model_str.txt").write_text(str(naive_automl.chosen_model))
            with open(artifacts_dir / "model.pkl", 'wb') as fh:
                pickle.dump(naive_automl.chosen_model, fh)
    except Exception:
        log.warning("Error when saving artifacts.", exc_info=True)



if __name__ == '__main__':
    call_run(run)
