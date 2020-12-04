import contextlib
import logging
import os
import psutil
import re
import shutil

import h2o
from h2o.automl import H2OAutoML

from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.datautils import to_data_frame, write_csv
from amlb.results import NoResultError, save_predictions_to_file
from amlb.utils import Monitoring, Timer, walk_apply, zip_path
from amlb.resources import config as rconfig
from frameworks.shared.callee import output_subdir

log = logging.getLogger(__name__)


class BackendMemoryMonitoring(Monitoring):

    def __init__(self, name=None, frequency_seconds=300, check_on_exit=False,
                 verbosity=0, log_level=logging.INFO):
        super().__init__(name, frequency_seconds, check_on_exit, "backend_monitoring_")
        self._verbosity = verbosity
        self._log_level = log_level

    def _check_state(self):
        sd = h2o.cluster().get_status_details()
        log.log(self._log_level, "System memory (bytes): %s", psutil.virtual_memory())
        log.log(self._log_level, "DKV: %s MB; Other: %s MB", sd['mem_value_size'][0] >> 20, sd['pojo_mem'][0] >> 20)


def run(dataset: Dataset, config: TaskConfig):
    log.info("\n**** H2O AutoML ****\n")
    # Mapping of benchmark metrics to H2O metrics
    metrics_mapping = dict(
        acc='mean_per_class_error',
        auc='AUC',
        logloss='logloss',
        mae='mae',
        mse='mse',
        r2='r2',
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
        jvm_memory = str(round(config.max_mem_size_mb * 2/3))+"M"   # leaving 1/3rd of available memory for XGBoost

        log.info("Starting H2O cluster with %s cores, %s memory.", nthreads, jvm_memory)
        max_port_range = 49151
        min_port_range = 1024
        rnd_port = os.getpid() % (max_port_range-min_port_range) + min_port_range
        port = config.framework_params.get('_port', rnd_port)

        h2o.init(nthreads=nthreads,
                 port=port,
                 min_mem_size=jvm_memory,
                 max_mem_size=jvm_memory,
                 strict_version_check=config.framework_params.get('_strict_version_check', True)
                 # log_dir=os.path.join(config.output_dir, 'logs', config.name, str(config.fold))
                 )

        # Load train as an H2O Frame, but test as a Pandas DataFrame
        log.debug("Loading train data from %s.", dataset.train.path)
        train = h2o.import_file(dataset.train.path, destination_frame=frame_name('train', config))
        # train.impute(method='mean')
        log.debug("Loading test data from %s.", dataset.test.path)
        test = h2o.import_file(dataset.test.path, destination_frame=frame_name('test', config))
        # test.impute(method='mean')

        log.info("Running model on task %s, fold %s.", config.name, config.fold)
        log.debug("Running H2O AutoML with a maximum time of %ss on %s core(s), optimizing %s.",
                  config.max_runtime_seconds, config.cores, sort_metric)

        aml = H2OAutoML(max_runtime_secs=config.max_runtime_seconds,
                        max_runtime_secs_per_model=round(config.max_runtime_seconds/2),  # to prevent timeout on ensembles
                        sort_metric=sort_metric,
                        seed=config.seed,
                        **training_params)

        monitor = (BackendMemoryMonitoring(frequency_seconds=rconfig().monitoring.frequency_seconds,
                                          check_on_exit=True,
                                          verbosity=rconfig().monitoring.verbosity) if config.framework_params.get('_monitor_backend', False)
                   # else contextlib.nullcontext  # Py 3.7+ only
                   else contextlib.contextmanager(iter)([0])
                   )
        with Timer() as training:
            with monitor:
                aml.train(y=dataset.target.index, training_frame=train)

        if not aml.leader:
            raise NoResultError("H2O could not produce any model in the requested time.")

        save_predictions(aml, test, dataset=dataset, config=config)
        save_artifacts(aml, dataset=dataset, config=config)

        return dict(
            models_count=len(aml.leaderboard),
            training_duration=training.duration
        )

    finally:
        if h2o.connection():
            # h2o.remove_all()
            h2o.connection().close()
        if h2o.connection().local_server:
            h2o.connection().local_server.shutdown()
        # if h2o.cluster():
        #     h2o.cluster().shutdown()


def frame_name(fr_type, config):
    return '_'.join([fr_type, config.name, str(config.fold)])


def save_artifacts(automl, dataset, config):
    artifacts = config.framework_params.get('_save_artifacts', ['leaderboard'])
    try:
        lb = automl.leaderboard.as_data_frame()
        log.debug("Leaderboard:\n%s", lb.to_string())
        if 'leaderboard' in artifacts:
            models_dir = output_subdir("models", config)
            write_csv(lb, os.path.join(models_dir, "leaderboard.csv"))
        if 'models' in artifacts:
            models_dir = output_subdir("models", config)
            all_models_se = next((mid for mid in lb['model_id'] if mid.startswith("StackedEnsemble_AllModels")),
                                 None)
            mformat = 'mojo' if 'mojos' in artifacts else 'json'
            if all_models_se and mformat == 'mojo':
                save_model(all_models_se, dest_dir=models_dir, mformat=mformat)
            else:
                for mid in lb['model_id']:
                    save_model(mid, dest_dir=models_dir, mformat=mformat)
                models_archive = os.path.join(models_dir, "models.zip")
                zip_path(models_dir, models_archive)

                def delete(path, isdir):
                    if path != models_archive and os.path.splitext(path)[1] in ['.json', '.zip']:
                        os.remove(path)
                walk_apply(models_dir, delete, max_depth=0)

        if 'models_predictions' in artifacts:
            predictions_dir = output_subdir("predictions", config)
            test = h2o.get_frame(frame_name('test', config))
            for mid in lb['model_id']:
                model = h2o.get_model(mid)
                save_predictions(model, test,
                                 dataset=dataset,
                                 config=config,
                                 predictions_file=os.path.join(predictions_dir, mid, 'predictions.csv'),
                                 preview=False
                                 )
            zip_path(predictions_dir,
                     os.path.join(predictions_dir, "models_predictions.zip"))

            def delete(path, isdir):
                if isdir:
                    shutil.rmtree(path, ignore_errors=True)
            walk_apply(predictions_dir, delete, max_depth=0)

        if 'logs' in artifacts:
            logs_dir = output_subdir("logs", config)
            h2o.download_all_logs(dirname=logs_dir)
    except Exception:
        log.debug("Error when saving artifacts.", exc_info=True)


def save_model(model_id, dest_dir='.', mformat='mojo'):
    model = h2o.get_model(model_id)
    if mformat == 'mojo':
        model.save_mojo(path=dest_dir)
        # model.download_mojo(path=dest_dir, get_genmodel_jar=True)
    else:
        model.save_model_details(path=dest_dir)


def save_predictions(model, test, dataset, config, predictions_file=None, preview=True):
    h2o_preds = model.predict(test).as_data_frame(use_pandas=False)
    preds = to_data_frame(h2o_preds[1:], columns=h2o_preds[0])
    y_pred = preds.iloc[:, 0]

    h2o_truth = test[:, dataset.target.index].as_data_frame(use_pandas=False, header=False)
    y_truth = to_data_frame(h2o_truth)

    predictions = y_pred.values
    probabilities = preds.iloc[:, 1:].values
    prob_labels = h2o_preds[0][1:]
    if all([re.fullmatch(r"p\d+", p) for p in prob_labels]):
        # for categories represented as numerical values, h2o prefixes the probabilities columns with p
        # in this case, we let the app setting the labels to avoid mismatch
        prob_labels = None
    truth = y_truth.values

    save_predictions_to_file(dataset=dataset,
                             output_file=config.output_predictions_file if predictions_file is None else predictions_file,
                             probabilities=probabilities,
                             probabilities_labels=prob_labels,
                             predictions=predictions,
                             truth=truth,
                             preview=preview)
