import logging
import os
import shutil
import warnings
import sys
import tempfile
warnings.simplefilter("ignore")

if sys.platform == 'darwin':
    os.environ['OMP_NUM_THREADS'] = '1'

import matplotlib
import pandas as pd
matplotlib.use('agg')  # no need for tk

from autogluon.tabular import TabularPredictor, TabularDataset
from autogluon.core.utils.savers import save_pd, save_pkl
import autogluon.core.metrics as metrics
from autogluon.tabular.version import __version__

from frameworks.shared.callee import call_run, result, output_subdir
from frameworks.shared.utils import Timer, zip_path

log = logging.getLogger(__name__)


def run(dataset, config):
    log.info(f"\n**** AutoGluon [v{__version__}] ****\n")
    try:
        from pip._internal.operations import freeze
        pip_dependencies = freeze.freeze()
        log.info('===== pip freeze =====')
        for p in pip_dependencies:
            log.info(p)
        log.info('======================\n')
    except:
        pass

    metrics_mapping = dict(
        acc=metrics.accuracy,
        auc=metrics.roc_auc,
        f1=metrics.f1,
        logloss=metrics.log_loss,
        mae=metrics.mean_absolute_error,
        mse=metrics.mean_squared_error,
        r2=metrics.r2,
        rmse=metrics.root_mean_squared_error,
    )

    perf_metric = metrics_mapping[config.metric] if config.metric in metrics_mapping else None
    if perf_metric is None:
        # TODO: figure out if we are going to blindly pass metrics through, or if we use a strict mapping
        log.warning("Performance metric %s not supported.", config.metric)

    is_classification = config.type == 'classification'
    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}

    train_path, test_path = dataset.train.path, dataset.test.path
    label = dataset.target.name
    problem_type = dataset.problem_type

    models_dir = tempfile.mkdtemp() + os.sep  # passed to AG

    with Timer() as training:
        predictor = TabularPredictor(
            label=label,
            eval_metric=perf_metric.name,
            path=models_dir,
            problem_type=problem_type,
        ).fit(
            train_data=train_path,
            time_limit=config.max_runtime_seconds,
            **training_params
        )

    test_data = TabularDataset(test_path)
    # Persist model in memory that is going to be predicting to get correct inference latency
    predictor.persist_models('best')

    if is_classification:
        with Timer() as predict:
            probabilities = predictor.predict_proba(test_data, as_multiclass=True)
        predictions = probabilities.idxmax(axis=1).to_numpy()
    else:
        with Timer() as predict:
            predictions = predictor.predict(test_data, as_pandas=False)
        probabilities = None

    prob_labels = probabilities.columns.values.astype(str).tolist() if probabilities is not None else None

    _leaderboard_extra_info = config.framework_params.get('_leaderboard_extra_info', False)  # whether to get extra model info (very verbose)
    _leaderboard_test = config.framework_params.get('_leaderboard_test', False)  # whether to compute test scores in leaderboard (expensive)
    leaderboard_kwargs = dict(silent=True, extra_info=_leaderboard_extra_info)
    # Disabled leaderboard test data input by default to avoid long running computation, remove 7200s timeout limitation to re-enable
    if _leaderboard_test:
        leaderboard_kwargs['data'] = test_data

    leaderboard = predictor.leaderboard(**leaderboard_kwargs)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        log.info(leaderboard)

    num_models_trained = len(leaderboard)
    if predictor._trainer.model_best is not None:
        num_models_ensemble = len(predictor._trainer.get_minimum_model_set(predictor._trainer.model_best))
    else:
        num_models_ensemble = 1

    save_artifacts(predictor, leaderboard, config, test_data=test_data)
    shutil.rmtree(predictor.path, ignore_errors=True)

    return result(output_file=config.output_predictions_file,
                  predictions=predictions,
                  probabilities=probabilities,
                  probabilities_labels=prob_labels,
                  target_is_encoded=False,
                  models_count=num_models_trained,
                  models_ensemble_count=num_models_ensemble,
                  training_duration=training.duration,
                  predict_duration=predict.duration)


def save_artifacts(predictor, leaderboard, config, test_data):
    artifacts = config.framework_params.get('_save_artifacts', ['leaderboard'])
    try:
        if 'leaderboard' in artifacts:
            leaderboard_dir = output_subdir("leaderboard", config)
            save_pd.save(path=os.path.join(leaderboard_dir, "leaderboard.csv"), df=leaderboard)

        if 'info' in artifacts:
            ag_info = predictor.info()
            info_dir = output_subdir("info", config)
            save_pkl.save(path=os.path.join(info_dir, "info.pkl"), object=ag_info)

        if 'models' in artifacts:
            shutil.rmtree(os.path.join(predictor.path, "utils"), ignore_errors=True)
            models_dir = output_subdir("models", config)
            zip_path(predictor.path, os.path.join(models_dir, "models.zip"))

        if 'zeroshot' in artifacts:
            zeroshot_dir = output_subdir("zeroshot", config)
            zeroshot_dict = get_zeroshot_artifact(predictor=predictor, test_data=test_data)
            save_pkl.save(path=os.path.join(zeroshot_dir, "zeroshot_metadata.pkl"), object=zeroshot_dict)
    except Exception:
        log.warning("Error when saving artifacts.", exc_info=True)


def get_zeroshot_artifact(predictor, test_data) -> dict:
    models = predictor.get_model_names(can_infer=True)

    pred_proba_dict_val = predictor.get_model_pred_proba_dict(inverse_transform=False, models=models)
    pred_proba_dict_test = predictor.get_model_pred_proba_dict(test_data, inverse_transform=False, models=models)

    val_data_source = 'train' if predictor._trainer.bagged_mode else 'val'
    _, y_val = predictor.load_data_internal(data=val_data_source, return_X=False, return_y=True)
    y_test = test_data[predictor.label]
    y_test = predictor.transform_labels(y_test, inverse=False)

    zeroshot_dict = dict(
        pred_proba_dict_val=pred_proba_dict_val,
        pred_proba_dict_test=pred_proba_dict_test,
        y_val=y_val,
        y_test=y_test,
        eval_metric=predictor.eval_metric.name,
        problem_type=predictor.problem_type,
        ordered_class_labels=predictor._learner.label_cleaner.ordered_class_labels,
        ordered_class_labels_transformed=predictor._learner.label_cleaner.ordered_class_labels_transformed,
        problem_type_transform=predictor._learner.label_cleaner.problem_type_transform,
        num_classes=predictor._learner.label_cleaner.num_classes,
        label=predictor.label,
    )

    return zeroshot_dict


if __name__ == '__main__':
    call_run(run)
