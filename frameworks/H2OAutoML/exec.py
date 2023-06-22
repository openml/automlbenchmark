import contextlib
import logging
import os
import pathlib

import psutil
import re

from packaging import version
import pandas as pd

import h2o
from h2o.automl import H2OAutoML

from frameworks.shared.callee import FrameworkError, call_run, output_subdir, result, \
    measure_inference_times
from frameworks.shared.utils import Monitoring, Namespace as ns, Timer, clean_dir, touch, zip_path

log = logging.getLogger(__name__)


class BackendMemoryMonitoring(Monitoring):

    def __init__(self, name=None, interval_seconds=300, check_on_exit=False,
                 verbosity=0, log_level=logging.INFO):
        super().__init__(name, interval_seconds, check_on_exit, "backend_monitoring_")
        self._verbosity = verbosity
        self._log_level = log_level

    def _check_state(self):
        sd = h2o.cluster().get_status_details()
        log.log(self._log_level, "System memory (bytes): %s", psutil.virtual_memory())
        log.log(self._log_level, "DKV: %s MB; Other: %s MB", sd['mem_value_size'][0] >> 20, sd['pojo_mem'][0] >> 20)


def run(dataset, config):
    log.info(f"\n**** H2O AutoML [v{h2o.__version__}] ****\n")
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

        init_params = config.framework_params.get('_init', {})
        if "logs" in config.framework_params.get('_save_artifacts', []):
            init_params['ice_root'] = output_subdir("logs", config)

        h2o.init(nthreads=nthreads,
                 port=port,
                 min_mem_size=jvm_memory,
                 max_mem_size=jvm_memory,
                 **init_params)

        import_kwargs = dict(escapechar='\\')
        # Load train as an H2O Frame, but test as a Pandas DataFrame
        log.debug("Loading train data from %s.", dataset.train.path)
        train = None
        if version.parse(h2o.__version__) >= version.parse("3.32.1"):  # previous versions may fail to parse correctly some rare arff files using single quotes as enum/string delimiters (pandas also fails on same datasets)
            import_kwargs['quotechar'] = '"'
            train = h2o.import_file(dataset.train.path, destination_frame=frame_name('train', config), **import_kwargs)
            if train.nlevels() != dataset.domains.cardinalities:
                h2o.remove(train)
                train = None
                import_kwargs['quotechar'] = "'"

        if not train:
            train = h2o.import_file(dataset.train.path, destination_frame=frame_name('train', config), **import_kwargs)
            # train.impute(method='mean')
        log.debug("Loading test data from %s.", dataset.test.path)
        test = h2o.import_file(dataset.test.path, destination_frame=frame_name('test', config), **import_kwargs)
        # test.impute(method='mean')

        if config.type == 'classification' and dataset.format == 'csv':
            train[dataset.target.index] = train[dataset.target.index].asfactor()
            test[dataset.target.index] = test[dataset.target.index].asfactor()

        log.info("Running model on task %s, fold %s.", config.name, config.fold)
        log.debug("Running H2O AutoML with a maximum time of %ss on %s core(s), optimizing %s.",
                  config.max_runtime_seconds, config.cores, sort_metric)

        aml = H2OAutoML(max_runtime_secs=config.max_runtime_seconds,
                        sort_metric=sort_metric,
                        seed=config.seed,
                        **training_params)

        monitor = (BackendMemoryMonitoring(interval_seconds=config.ext.monitoring.interval_seconds,
                                           check_on_exit=True,
                                           verbosity=config.ext.monitoring.verbosity)
                   if config.framework_params.get('_monitor_backend', False)
                   else contextlib.nullcontext()  # Py 3.7+ only
                   # else contextlib.contextmanager(lambda: (_ for _ in (0,)))()
                   )
        with Timer() as training:
            with monitor:
                aml.train(y=dataset.target.index, training_frame=train)
        log.info(f"Finished fit in {training.duration}s.")


        if not aml.leader:
            raise FrameworkError("H2O could not produce any model in the requested time.")

        def infer(path: str):
            filename = pathlib.Path(path).name
            # H2O can't do inference on single row arff, it needs columns explicitly:
            # https://github.com/h2oai/h2o-3/issues/15572
            batch = h2o.import_file(path, col_names=train.names, destination_frame=frame_name(filename, config), **import_kwargs)
            return aml.predict(batch)

        inference_times = {}
        if config.measure_inference_time:
            inference_times["file"] = measure_inference_times(infer, dataset.inference_subsample_files)
            log.info(f"Finished inference time measurements.")

        with Timer() as predict:
            preds = aml.predict(test)
        log.info(f"Finished predict in {predict.duration}s.")

        preds = extract_preds(preds, test, dataset=dataset)
        save_artifacts(aml, dataset=dataset, config=config)

        return result(
            output_file=config.output_predictions_file,
            predictions=preds.predictions,
            truth=preds.truth,
            probabilities=preds.probabilities,
            probabilities_labels=preds.probabilities_labels,
            models_count=len(aml.leaderboard),
            training_duration=training.duration,
            predict_duration=predict.duration,
            inference_times=inference_times,
        )

    finally:
        con = h2o.connection()
        if con:
            # h2o.remove_all()
            con.close()
            if con.local_server:
                con.local_server.shutdown()
        # if h2o.cluster():
        #     h2o.cluster().shutdown()


def frame_name(fr_type, config):
    return '_'.join([fr_type, config.name, str(config.fold)])


def save_artifacts(automl, dataset, config):
    artifacts = config.framework_params.get('_save_artifacts', ['leaderboard'])
    try:
        models_artifacts = []
        lb_pat = re.compile(r"leaderboard(?:\[(.*)\])?")
        lb_match = next((lb_pat.fullmatch(a) for a in artifacts), None)
        if lb_match:
            lb_ext = list(filter(None, re.split("[,; ]", (lb_match.group(1) or ""))))
            lb = h2o.automl.get_leaderboard(automl, lb_ext).as_data_frame()
            models_dir = output_subdir("models", config)
            lb_path = os.path.join(models_dir, "leaderboard.csv")
            write_csv(lb, lb_path)
            models_artifacts.append(lb_path)
        else:
            lb = automl.leaderboard.as_data_frame()
        log.debug("Leaderboard:\n%s", lb.to_string())

        models_pat = re.compile(r"models(\[(json|binary|mojo)(?:,(\d+))?\])?")
        models = list(filter(models_pat.fullmatch, artifacts))
        for m in models:
            models_dir = output_subdir("models", config)
            all_models_se = next((mid for mid in lb['model_id'] if mid.startswith("StackedEnsemble_AllModels")),
                                 None)
            match = models_pat.fullmatch(m)
            mformat = match.group(2) or 'json'
            topN = int(match.group(3) or -1)
            if topN < 0 and mformat != 'json' and all_models_se:
                models_artifacts.append(save_model(all_models_se, dest_dir=models_dir, mformat=mformat))
            else:
                count = 0
                for mid in lb['model_id']:
                    if topN < 0 or count < topN:
                        save_model(mid, dest_dir=models_dir, mformat=mformat)
                        count += 1
                    else:
                        break

                models_archive = os.path.join(models_dir, f"models_{mformat}.zip")
                zip_path(models_dir, models_archive, filter_=lambda p: p not in models_artifacts)
                models_artifacts.append(models_archive)
                clean_dir(models_dir,
                          filter_=lambda p: p not in models_artifacts
                                          and os.path.splitext(p)[1] in ['.json', '.zip', ''])

        if 'model_predictions' in artifacts:
            predictions_dir = output_subdir("predictions", config)
            test = h2o.get_frame(frame_name('test', config))
            for mid in lb['model_id']:
                model = h2o.get_model(mid)
                h2o_preds = model.predict(test)
                preds = extract_preds(h2o_preds, test, dataset=dataset)
                if preds.probabilities_labels is None:
                    preds.probabilities_labels = preds.h2o_labels
                write_preds(preds, os.path.join(predictions_dir, mid, 'predictions.csv'))
            predictions_zip = os.path.join(predictions_dir, "model_predictions.zip")
            zip_path(predictions_dir, predictions_zip)
            clean_dir(predictions_dir, filter_=lambda p: os.path.isdir(p))

        if 'logs' in artifacts:
            logs_dir = output_subdir("logs", config)
            logs_zip = os.path.join(logs_dir, "h2o_logs.zip")
            zip_path(logs_dir, logs_zip)
            # h2o.download_all_logs(dirname=logs_dir)
            clean_dir(logs_dir, filter_=lambda p: p != logs_zip)
            write_csv(automl.event_log.as_data_frame(), os.path.join(logs_dir, 'aml_event_log.csv'))
    except Exception:
        log.debug("Error when saving artifacts.", exc_info=True)


def save_model(model_id, dest_dir='.', mformat='json'):
    model = h2o.get_model(model_id)
    if mformat == 'mojo':
        return model.save_mojo(path=dest_dir)
        # model.download_mojo(path=dest_dir, get_genmodel_jar=True)
    elif mformat == 'binary':
        return h2o.save_model(model, path=dest_dir)
        # return h2o.download_model(model, path=dest_dir)
    else:
        return model.save_model_details(path=dest_dir)


def extract_preds(h2o_preds, test, dataset, ):
    h2o_preds = h2o_preds.as_data_frame(use_pandas=False)
    preds = to_data_frame(h2o_preds[1:], columns=h2o_preds[0])
    y_pred = preds.iloc[:, 0]

    h2o_truth = test[:, dataset.target.index].as_data_frame(use_pandas=False, header=False)
    y_truth = to_data_frame(h2o_truth)

    predictions = y_pred.values
    probabilities = preds.iloc[:, 1:].values
    prob_labels = h2o_labels = h2o_preds[0][1:]
    if all([re.fullmatch(r"p(-?\d)+", p) for p in prob_labels]):
        # for categories represented as numerical values, h2o prefixes the probabilities columns with p
        # in this case, we let the app setting the labels to avoid mismatch
        prob_labels = None
    truth = y_truth.values

    return ns(predictions=predictions,
              truth=truth,
              probabilities=probabilities,
              probabilities_labels=prob_labels,
              h2o_labels=h2o_labels)


def write_preds(preds, path):
    df = to_data_frame(preds.probabilities, columns=preds.probabilities_labels)
    df = df.assign(predictions=preds.predictions)
    df = df.assign(truth=preds.truth)
    write_csv(df,  path)


def to_data_frame(arr, columns=None):
    return pd.DataFrame.from_records(arr, columns=columns)


def write_csv(df, path):
    touch(path)
    df.to_csv(path, header=True, index=False)


if __name__ == '__main__':
    call_run(run)

