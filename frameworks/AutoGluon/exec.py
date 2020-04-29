import logging
import os
import warnings
warnings.simplefilter("ignore")

import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('agg')  # no need for tk

from autogluon.task.tabular_prediction.tabular_prediction import TabularPrediction as task
from autogluon.utils.tabular.utils.savers import save_pd, save_pkl
import autogluon.utils.tabular.metrics as metrics

from frameworks.shared.callee import call_run, result, Timer, touch

log = logging.getLogger(__name__)


def run(dataset, config):
    log.info("\n**** AutoGluon ****\n")

    metrics_mapping = dict(
        acc=metrics.accuracy,
        auc=metrics.roc_auc,
        f1=metrics.f1,
        logloss=metrics.log_loss,
        mae=metrics.mean_absolute_error,
        mse=metrics.mean_squared_error,
        r2=metrics.r2
    )

    perf_metric = metrics_mapping[config.metric] if config.metric in metrics_mapping else None
    if perf_metric is None:
        # TODO: figure out if we are going to blindly pass metrics through, or if we use a strict mapping
        log.warning("Performance metric %s not supported.", config.metric)

    is_classification = config.type == 'classification'

    train = pd.DataFrame(dataset.train.data, columns=dataset.columns)
    label = dataset.target.name

    output_dir = make_subdir("models", config)
    with Timer() as training:
        predictor = task.fit(
            train_data=train,
            label=label,
            output_directory=output_dir,
            time_limits=config.max_runtime_seconds,
            eval_metric=perf_metric.name,
        )

    test = pd.DataFrame(dataset.test.data, columns=dataset.columns)
    X_test = test.drop(columns=label)
    y_test = test[label]

    with Timer() as predict:
        predictions = predictor.predict(X_test)

    probabilities = predictor.predict_proba(X_test, as_pandas=True) if is_classification else None
    if is_classification and len(probabilities.shape) == 1:
        # for binary, AutoGluon returns only one probability and it's not systematically the smaller or greater label.
        # Here, we're trying to identify to which label correspond the probabilities
        # and from there we can build the probabilities for all labels.
        classes = dataset.target.classes
        prob_class = next((predictions[i] if p > 0.5 else next(c for c in classes if c != predictions[i])
                           for i, p in enumerate(probabilities) if p != 0.5),
                          classes[1])
        probabilities = (np.array([[1-row, row] for row in probabilities]) if prob_class == classes[1]
                         else np.array([[row, 1-row] for row in probabilities]))

    leaderboard = predictor._learner.leaderboard(X_test, y_test, silent=True)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(leaderboard)

    save_artifacts(predictor, leaderboard, config)

    num_models_trained = len(leaderboard)
    num_models_ensemble = len(leaderboard[leaderboard['stack_level'] > 0])

    return result(output_file=config.output_predictions_file,
                  predictions=predictions,
                  truth=y_test,
                  probabilities=probabilities,
                  target_is_encoded=False,
                  models_count=num_models_trained,
                  models_ensemble_count=num_models_ensemble,
                  training_duration=training.duration,
                  predict_duration=predict.duration)


def make_subdir(name, config):
    subdir = os.path.join(config.output_dir, name, config.name, str(config.fold))
    touch(subdir, as_dir=True)
    return subdir


def save_artifacts(predictor, leaderboard, config):
    artifacts = config.framework_params.get('_save_artifacts', ['leaderboard'])
    try:
        if 'leaderboard' in artifacts:
            models_dir = make_subdir("models", config)
            save_pd.save(path=os.path.join(models_dir, "leaderboard.csv"), df=leaderboard)

        if 'info' in artifacts:
            ag_info = predictor._learner.get_info()
            info_dir = make_subdir("info", config)
            save_pkl.save(path=os.path.join(info_dir, "info.pkl"), object=ag_info)
    except:
        log.debug("Error when saving artifacts.", exc_info=True)


if __name__ == '__main__':
    call_run(run)
