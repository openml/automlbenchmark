import logging
import os
import pickle
import warnings

import matplotlib
import numpy as np
matplotlib.use("agg")  # no need for tk

from lightautoml.tasks import Task
from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
from lightautoml import __version__

from frameworks.shared.callee import call_run, result, output_subdir
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

    X_test, y_test = dataset.test.X, dataset.test.y
    log.info("Predicting on the test set...")
    with Timer() as predict:
        preds = automl.predict(X_test).data

    if is_classification:
        probabilities = preds

        if dataset.problem_type == 'binary':
            probabilities = np.vstack([
                1 - probabilities[:, 0], probabilities[:, 0]
            ]).T

        predictions = np.argmax(probabilities, axis=1)

    else:
        probabilities = None
        predictions = preds

    log.debug(probabilities)
    log.debug(config.output_predictions_file)

    save_artifacts(automl, config)

    return result(
        output_file=config.output_predictions_file,
        probabilities=probabilities,
        predictions=predictions,
        target_is_encoded=is_classification,
        training_duration=training.duration,
        predict_duration=predict.duration,
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
