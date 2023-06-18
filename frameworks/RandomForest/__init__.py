from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.utils import call_script_in_same_dir, unsparsify


def setup(*args, **kwargs):
    call_script_in_same_dir(__file__, "setup.sh", *args, **kwargs)


def run(dataset: Dataset, config: TaskConfig):
    from amlb.datautils import impute_array
    from frameworks.shared.caller import run_in_venv

    encode = config.framework_params.get('_encode', True)
    X_train, X_test = impute_array(dataset.train.X_enc, dataset.test.X_enc) if encode else (dataset.train.X, dataset.test.X)
    y_train, y_test = (dataset.train.y_enc, dataset.test.y_enc) if encode else (dataset.train.y, dataset.test.y)
    y_train, y_test = unsparsify(y_train, y_test)
    data = dict(
        train=dict(
            X=X_train,
            y=y_train
        ),
        test=dict(
            X=X_test,
            y=y_test
        ),
    )
    if config.measure_inference_time:
        data["inference_subsample_files"] = dataset.inference_subsample_files(fmt="parquet", scikit_safe=True)

    return run_in_venv(__file__, "exec.py",
                       input_data=data, dataset=dataset, config=config)

