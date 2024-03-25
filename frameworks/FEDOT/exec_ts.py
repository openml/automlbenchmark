import logging
import os
from pathlib import Path
import numpy as np

from fedot.api.main import Fedot
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum

from frameworks.shared.callee import call_run, result, output_subdir
from frameworks.shared.utils import Timer, load_timeseries_dataset

log = logging.getLogger(__name__)


def run(dataset, config):
    log.info("\n**** FEDOT ****\n")

    scoring_metric = get_fedot_metrics(config)

    training_params = {"preset": "best_quality", "n_jobs": config.cores}
    training_params.update({k: v for k, v in config.framework_params.items() if not k.startswith('_')})
    n_jobs = training_params["n_jobs"]

    log.info(f"Running FEDOT with a maximum time of {config.max_runtime_seconds}s on {n_jobs} cores, \
             optimizing {scoring_metric}")
    runtime_min = config.max_runtime_seconds / 60

    task = Task(
        TaskTypesEnum.ts_forecasting,
        TsForecastingParams(forecast_length=dataset.forecast_horizon_in_steps)
    )

    train_df, test_df = load_timeseries_dataset(dataset)
    id_column = dataset.id_column

    log.info('Predicting on the test set.')
    training_duration, predict_duration = 0, 0
    models_count = 0
    truth_only = test_df[dataset.target].values
    predictions = []


    for label, ts in train_df.groupby(id_column, sort=False):
        train_series = ts[dataset.target].to_numpy()

    for label in train_df[id_column].unique():
        train_sub_df = train_df[train_df[id_column] == label].drop(columns=[id_column], axis=1)
        train_series = np.array(train_sub_df[dataset.target])
        train_input = InputData(
            idx=train_sub_df.index.to_numpy(),
            features=train_series,
            target=train_series,
            task=task,
            data_type=DataTypesEnum.ts
        )

        test_sub_df = test_df[test_df[id_column] == label].drop(columns=[id_column], axis=1)
        test_series = np.array(test_sub_df[dataset.target])
        test_input = InputData(
            idx=test_sub_df.index.to_numpy(),
            features=train_series,
            target=test_series,
            task=task,
            data_type=DataTypesEnum.ts
        )

        fedot = Fedot(
            problem=TaskTypesEnum.ts_forecasting.value,
            task_params=task.task_params,
            timeout=runtime_min,
            metric=scoring_metric,
            seed=config.seed,
            max_pipeline_fit_time=runtime_min / 10,
            **training_params
        )

        with Timer() as training:
            fedot.fit(train_input)
        training_duration += training.duration

        with Timer() as predict:
            try:
                prediction = fedot.predict(test_input)
            except Exception as e:
                log.info('Pipeline crashed. Using no-op forecasting')
                prediction = np.full(len(test_series), train_series[-1])

        predict_duration += predict.duration

        predictions.append(prediction)
        models_count += fedot.current_pipeline.length

    save_artifacts(fedot, config)
    return result(output_file=config.output_predictions_file,
                  predictions=np.hstack(predictions),
                  truth=truth_only,
                  target_is_encoded=False,
                  models_count=models_count,
                  training_duration=training_duration,
                  predict_duration=predict_duration)


def get_fedot_metrics(config):
    metrics_mapping = dict(
        mape='mape',
        smape='smape',
        mase='mase',
        mse='mse',
        rmse='rmse',
        mae='mae',
        r2='r2',
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
