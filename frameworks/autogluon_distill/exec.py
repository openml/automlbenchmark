import logging

import pandas as pd
import numpy as np

from autogluon.task.tabular_prediction.tabular_prediction import TabularPrediction as task
from autogluon.utils.tabular.utils.loaders import load_pd
from autogluon.utils.tabular.utils.savers import save_pd, save_pkl
import autogluon.utils.tabular.metrics as metrics

from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.results import save_predictions_to_file
from amlb.utils import Timer


log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    log.info("\n**** AutoGluon ****\n")

    print('#################')
    print('Config:')
    print(config.__json__())
    print()
    print('Dataset:')
    print(dataset.__dict__)
    print('#################')

    if config.fold != 0:
        raise AssertionError('config.fold should only be 0 when running AutoGluon distill! Value: %s' % config.fold)

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

    X_train = dataset.train.X
    y_train = dataset.train.y
    X_test = dataset.test.X
    y_test = dataset.test.y

    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)

    train_path = 'tmp/tmp_file_train.csv'
    test_path = 'tmp/tmp_file_test.csv'

    save_pd.save(path=train_path, df=X_train)
    save_pd.save(path=test_path, df=X_test)
    del X_train
    del X_test

    # Save and load data to remove any pre-set dtypes, we want to observe performance from worst-case scenario: raw csv

    X_train = load_pd.load(path=train_path)
    X_train['__label__'] = y_train

    with Timer() as training:
        predictor = task.fit(
            train_data=X_train,
            label='__label__',
            output_directory='tmp/',
            time_limits=config.max_runtime_seconds,
            eval_metric=perf_metric.name,
            auto_stack=True,
            verbosity=2,
            enable_fit_continuation=True,
        )

    predictor = task.load('tmp/', verbosity=4)  # use high-verbosity to see distillation process details.
    # Distill ensemble into single model:
    learner = predictor._learner

    print('STARTING DISTILLATION')

    # distill_time_limits = 2 * config.max_runtime_seconds
    # num_augmented_samples = max(100000, 2 * len(X_train))  # distillation-training will take longer the bigger this value is, but bigger values can produce superior distilled models.
    # learner.augment_distill(num_augmented_samples=num_augmented_samples, time_limits=distill_time_limits)

    # Compare best compressed single model with best distilled model:
    trainer = learner.load_trainer()
    best_baggedbase_model = trainer.best_single_model(stack_name='core', stack_level=0)
    best_compressed_model = learner.refit_single_full(models=[best_baggedbase_model])[0]
    # best_distilled_model = trainer.best_single_model(stack_name='distill', stack_level=0)
    # print("Best compressed: %s, best distill: %s" % (best_compressed_model, best_distilled_model))
    print("Best compressed: %s" % best_compressed_model)

    X_test = load_pd.load(path=test_path)
    with Timer() as predicting:
        predictions = predictor.predict(X_test, model=best_compressed_model)

    probabilities = predictor.predict_proba(X_test, model=best_compressed_model) if is_classification else None
    if is_classification & (len(probabilities.shape) == 1):
        probabilities = np.array([[1-row, row] for row in probabilities])

    num_models_trained = len(predictor._trainer.get_model_names_all())
    num_models_ensemble = num_models_trained  # TODO: Fixme, not accurate

    leaderboard = predictor._learner.leaderboard(X_test, y_test, silent=True)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(leaderboard)
    ag_info = predictor._learner.get_info()
    print(ag_info)

    output_dir = config.output_dir
    ag_model_scores_path = output_dir + '/' + 'ag_scores/' + 'scores_' + str(config.fold) + '.csv'
    save_pd.save(path=ag_model_scores_path, df=leaderboard)

    ag_info_path = output_dir + '/' + 'ag_info/' + 'info_' + str(config.fold) + '.pkl'
    save_pkl.save(path=ag_info_path, object=ag_info)

    save_predictions_to_file(dataset=dataset,
                             output_file=config.output_predictions_file,
                             probabilities=probabilities,
                             predictions=predictions,
                             truth=y_test,
                             target_is_encoded=False)

    return dict(
        models_count=num_models_trained,
        models_ensemble_count=num_models_ensemble,
        training_duration=training.duration,
        predict_duration=predicting.duration,
    )
