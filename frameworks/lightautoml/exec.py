import logging
import os

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import log_loss, roc_auc_score

from frameworks.shared.callee import call_run, result, output_subdir, utils
from lightautoml.tasks import Task
from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML

log = logging.getLogger(__name__)


def run(dataset, config):
    log.info("\n**** lightautoml (R) ****\n")
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=DeprecationWarning)

    is_classification = config.type == 'classification'

    y_train, y_test = dataset.train.y_enc, dataset.test.y_enc

    column_names, _ = zip(*dataset.columns)
    column_types = dict(dataset.columns)
    label = dataset.target.name

    df_train = pd.DataFrame(dataset.train.data, columns=column_names).astype(column_types, copy=False)
    df_train[dataset.target.name] = y_train

    task = Task(dataset.problem_type)
    automl = TabularUtilizedAutoML(task=task, timeout=config.max_runtime_seconds, random_state=config.seed)

    with utils.Timer() as training:
        oof_pred = automl.fit_predict(train_data=df_train, roles={'target': label}).data

    log.info("Predicting on the test set.")

    df_test = pd.DataFrame(dataset.test.data, columns=column_names).astype(column_types, copy=False)
    df_x_test = df_test.drop(columns=label)

    with utils.Timer() as predict:
        probabilities = automl.predict(df_x_test).data

    if probabilities is not None:
        if dataset.problem_type == 'binary':
            probabilities = np.vstack([
                1 - probabilities[:, 0], probabilities[:, 0]
            ]).T

    if dataset.problem_type == 'binary':
        oof_pred = oof_pred[:, 0]
        flags = ~np.isnan(oof_pred)
        y_oof = y_train[flags]

        oof_score = roc_auc_score(
            y_oof,
            oof_pred[flags]
        )
    elif dataset.problem_type == 'multiclass':
        flags = (np.isnan(oof_pred).sum(axis=1) == 0)

        oof_score = log_loss(
            y_train[flags],
            oof_pred[flags, :]
        )

    log.debug(probabilities)
    log.debug(config.output_predictions_file)
    print('OOF score: {}'.format(oof_score))

    return result(
        output_file=config.output_predictions_file,
        predictions=np.argmax(probabilities, axis=1),
        truth=y_test,
        probabilities=probabilities,
        target_is_encoded=is_classification,
        training_duration=training.duration,
        predict_duration=predict.duration,
    )


def save_artifacts(automl, config):
    try:
        artifacts = config.framework_params.get('_save_artifacts', False)
        models_dir = output_subdir("models", config)

        if 'models' in artifacts:
            with open(os.path.join(models_dir, 'automl.pickle'), 'wb') as f:
                pickle.dump(automl, f)

    except Exception:
        log.warning("Error when saving artifacts.", exc_info=True)


if __name__ == '__main__':
    call_run(run)

