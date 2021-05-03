from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.datautils import reorder_dataset
from amlb.utils import call_script_in_same_dir


def setup(*args, **kwargs):
    call_script_in_same_dir(__file__, "setup.sh", *args, **kwargs)


def run(dataset: Dataset, config: TaskConfig):
    from frameworks.shared.caller import run_in_venv

    train_path = dataset.train.path
    test_path = dataset.test.path
    backend = config.framework_params.get('_backend')
    # ML-Plan requires the target attribute to be the last column
    if dataset.target.index != len(dataset.predictors):
        train_path = reorder_dataset(dataset.train.path, target_src=dataset.target.index)
        test_path = reorder_dataset(dataset.test.path, target_src=dataset.target.index)

    data = dict(
        train=dict(path=train_path),
        test=dict(path=test_path),
        target=dict(index=dataset.target.index),
    )

    return run_in_venv(__file__, "exec.py",
                       input_data=data, dataset=dataset, config=config)
