from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.utils import call_script_in_same_dir, unsparsify


def setup(*args, **kwargs):
    call_script_in_same_dir(__file__, "setup.sh", *args, **kwargs)


def run(dataset: Dataset, config: TaskConfig):
    from frameworks.shared.caller import run_in_venv

    data = dict(
        target=dataset.target.name,
        train=dict(
            X=dataset.train.X,
            y=unsparsify(dataset.train.y_enc),
        ),
        test=dict(
            X=dataset.test.X,
            y=unsparsify(dataset.test.y_enc),
        ),
    )
    if config.measure_inference_time:
        data["inference_subsample_files"] = dataset.inference_subsample_files(fmt="parquet")
    options = dict(
        serialization=dict(sparse_dataframe_deserialized_format='dense')
    )

    return run_in_venv(__file__, "exec.py",
                       input_data=data, dataset=dataset, config=config, options=options)

