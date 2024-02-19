import logging
import os
from pathlib import Path

from fedot.api.main import Fedot
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.data.data import InputData

from frameworks.shared.callee import call_run, result, output_subdir
from frameworks.shared.utils import Timer
from joblib.externals.loky import get_reusable_executor

log = logging.getLogger(__name__)


def run(dataset, config):
    log.info("\n**** FEDOT ****\n")

    scoring_metric = get_fedot_metrics(config)

    training_params = {"preset": "best_quality", "n_jobs": config.cores}
    training_params |= {k: v for k, v in config.framework_params.items() if not k.startswith('_')}
    n_jobs = training_params["n_jobs"]

    log.info('Running FEDOT with a maximum time of %ss on %s cores, optimizing %s.',
             config.max_runtime_seconds, n_jobs, scoring_metric)
    runtime_min = config.max_runtime_seconds / 60

    task = Task(
        TaskTypesEnum.ts_forecasting,
        TsForecastingParams(forecast_length=dataset.forecast_horizon_in_steps)
    )
    train_input = InputData.from_csv_time_series(
        file_path=dataset.train_path,
        task=task,
        target_column=dataset.target,
        index_col=dataset.timestamp_column
    )
    test_input = InputData.from_csv_time_series(
        file_path=dataset.test_path,
        task=task,
        target_column=dataset.target,
        index_col=dataset.timestamp_column
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

    log.info('Predicting on the test set.')
    with Timer() as predict:
        predictions = fedot.predict(test_input)

    save_artifacts(fedot, config)
    get_reusable_executor().shutdown(wait=True)

    return result(output_file=config.output_predictions_file,
                  predictions=predictions,
                  truth=test_input.target,
                  target_is_encoded=False,
                  models_count=fedot.current_pipeline.length,
                  training_duration=training.duration,
                  predict_duration=predict.duration)


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
        log.warning("Performance metric %s not supported.", config.metric)

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
