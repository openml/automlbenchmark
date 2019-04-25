import logging
import os

import h2o
from h2o.automl import H2OAutoML

from automl.benchmark import TaskConfig
from automl.data import Dataset
from automl.datautils import to_data_frame, write_csv
from automl.results import NoResultError, save_predictions_to_file
from automl.utils import Timer, split_path, path_from_split

log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    log.info("\n**** H2O AutoML ****\n")
    # Mapping of benchmark metrics to H2O metrics
    metrics_mapping = dict(
        acc='mean_per_class_error',
        auc='AUC',
        logloss='logloss',
        mae='mae',
        mse='mse',
        rmse='rmse',
        rmsle='rmsle'
    )
    sort_metric = metrics_mapping[config.metric] if config.metric in metrics_mapping else None
    if sort_metric is None:
        # TODO: Figure out if we are going to blindly pass metrics through, or if we use a strict mapping
        log.warning("Performance metric %s not supported, defaulting to AUTO.", config.metric)

    try:
        training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}
        nthreads = config.framework_params.get('_nthreads', config.cores)

        log.info("Starting H2O cluster with %s cores, %sMB memory.", nthreads, config.max_mem_size_mb)
        h2o.init(nthreads=nthreads,
                 min_mem_size=str(config.max_mem_size_mb)+"M",
                 max_mem_size=str(config.max_mem_size_mb)+"M",
                 log_dir=os.path.join(config.output_dir, 'logs', config.name, str(config.fold)))

        # Load train as an H2O Frame, but test as a Pandas DataFrame
        log.debug("Loading train data from %s.", dataset.train.path)
        train = h2o.import_file(dataset.train.path)
        # train.impute(method='mean')
        log.debug("Loading test data from %s.", dataset.test.path)
        test = h2o.import_file(dataset.test.path)
        # test.impute(method='mean')

        log.info("Running model on task %s, fold %s.", config.name, config.fold)
        log.debug("Running H2O AutoML with a maximum time of %ss on %s core(s), optimizing %s.",
                  config.max_runtime_seconds, config.cores, sort_metric)

        aml = H2OAutoML(max_runtime_secs=config.max_runtime_seconds,
                        sort_metric=sort_metric,
                        seed=config.seed,
                        **training_params)

        with Timer() as training:
            aml.train(y=dataset.target.index, training_frame=train)

        if not aml.leader:
            raise NoResultError("H2O could not produce any model in the requested time.")

        lb = aml.leaderboard.as_data_frame()
        log.debug("Leaderboard:\n%s", lb.to_string())
        lbf = split_path(config.output_predictions_file)
        lbf.extension = '.leaderboard.csv'
        lbf = path_from_split(lbf)
        write_csv(lb, lbf)

        h2o_preds = aml.predict(test).as_data_frame(use_pandas=False)
        preds = to_data_frame(h2o_preds[1:], columns=h2o_preds[0])
        y_pred = preds.iloc[:, 0]

        h2o_truth = test[:, dataset.target.index].as_data_frame(use_pandas=False, header=False)
        y_truth = to_data_frame(h2o_truth)

        predictions = y_pred.values
        probabilities = preds.iloc[:, 1:].values
        truth = y_truth.values

        save_predictions_to_file(dataset=dataset,
                                 output_file=config.output_predictions_file,
                                 probabilities=probabilities,
                                 predictions=predictions,
                                 truth=truth)

        return dict(
            models_count=len(aml.leaderboard),
            training_duration=training.duration
        )

    finally:
        if h2o.connection():
            h2o.remove_all()
            h2o.connection().close()
        if h2o.connection().local_server:
            h2o.connection().local_server.shutdown()
        # if h2o.cluster():
        #     h2o.cluster().shutdown()

