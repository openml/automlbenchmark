from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.utils import call_script_in_same_dir, unsparsify


def setup(*args, **kwargs):
    call_script_in_same_dir(__file__, "setup.sh", *args, **kwargs)


def run(dataset: Dataset, config: TaskConfig):
    from frameworks.shared.caller import run_in_venv

    data = dict(
        train=dict(
            X=dataset.train.X,
            y=dataset.train.y,
            X_enc=dataset.train.X_enc,
            y_enc=unsparsify(dataset.train.y_enc),
        ),
        test=dict(
            X=dataset.test.X,
            y=dataset.test.y,
            X_enc=dataset.test.X_enc,
            y_enc=unsparsify(dataset.test.y_enc),
        ),
        predictors_type=['Numerical' if p.is_numerical() else 'Categorical' for p in dataset.predictors],
    )
    if config.measure_inference_time:
        data["inference_subsample_files"] = dataset.inference_subsample_files(fmt="parquet")

    return run_in_venv(__file__, "exec.py",
                       input_data=data, dataset=dataset, config=config)

