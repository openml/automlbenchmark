import logging
import os
import pickle
import warnings
from typing import Union

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("agg")  # no need for tk

from lightautoml.tasks import Task
from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
from lightautoml import __version__

from frameworks.shared.callee import call_run, result, output_subdir, \
    measure_inference_times
from frameworks.shared.utils import Timer

log = logging.getLogger(__name__)


def run(dataset, config):
    log.info(f"\n**** lightautoml (R) [{__version__}] ****\n")

    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=DeprecationWarning)

    is_classification = config.type == 'classification'

    label = dataset.target.name
    df_train = dataset.train.data

    max_mem_size_gb = float(config.max_mem_size_mb) / 1024
    task = Task(dataset.problem_type if dataset.problem_type != 'regression' else 'reg')
    automl = TabularUtilizedAutoML(task=task, timeout=config.max_runtime_seconds, cpu_limit=config.cores,
                                   memory_limit=max_mem_size_gb, random_state=config.seed)

    log.info("Training...")
    with Timer() as training:
        automl.fit_predict(train_data=df_train, roles={'target': label})
    log.info(f"Finished fit in {training.duration}s.")

    def infer(data: Union[str, pd.DataFrame]):
        batch = pd.read_parquet(data) if isinstance(data, str) else data
        return automl.predict(batch)

    inference_times = {}
    if config.measure_inference_time:
        inference_times["file"] = measure_inference_times(infer, dataset.inference_subsample_files)
        inference_times["df"] = measure_inference_times(
            infer,
            [(1, dataset.test.X.sample(1, random_state=i)) for i in range(100)],
        )
    log.info(f"Finished inference time measurements.")


    log.info("Predicting on the test set...")
    with Timer() as predict:
        X_test, y_test = dataset.test.X, dataset.test.y
        preds = automl.predict(X_test).data

    probabilities_labels = None
    if is_classification:
        probabilities = preds

        if dataset.problem_type == 'binary':
            probabilities = np.vstack([
                1 - probabilities[:, 0], probabilities[:, 0]
            ]).T

        predictions = np.argmax(probabilities, axis=1)
        class_map = automl.outer_pipes[0].ml_algos[0].models[0][0].reader.class_mapping
        if class_map is None and df_train[label].dtype == bool:
            class_map = {False: 0, True: 1}
        if class_map:
            column_to_class = {col: class_ for class_, col in class_map.items()}
            predictions = list(map(column_to_class.get, predictions))
            probabilities_labels = [column_to_class[col] for col in sorted(column_to_class)]
    else:
        probabilities = None
        predictions = preds

    log.debug(probabilities)
    log.debug(config.output_predictions_file)
    log.info(f"Finished predict in {predict.duration}s.")


    save_artifacts(automl, config)

    return result(
        output_file=config.output_predictions_file,
        probabilities_labels=probabilities_labels,
        probabilities=probabilities,
        predictions=predictions,
        training_duration=training.duration,
        predict_duration=predict.duration,
        inference_times=inference_times,
    )


def save_artifacts(automl, config):
    try:
        artifacts = config.framework_params.get('_save_artifacts', [])
        models_dir = output_subdir("models", config)

        if 'models' in artifacts:
            with open(os.path.join(models_dir, 'automl.pickle'), 'wb') as f:
                pickle.dump(automl, f)

    except Exception:
        log.warning("Error when saving artifacts.", exc_info=True)


if __name__ == '__main__':
    call_run(run)
