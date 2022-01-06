from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.datautils import Encoder, impute_array
from amlb.utils import call_script_in_same_dir


def setup(*args, **kwargs):
    call_script_in_same_dir(__file__, "setup.sh", *args, **kwargs)


def run(dataset: Dataset, config: TaskConfig):
    from frameworks.shared.caller import run_in_venv

    X_train, X_test = impute_array(dataset.train.X_enc, dataset.test.X_enc)
    y_train, y_test = dataset.train.y_enc, dataset.test.y_enc
    data = dict(
        train=dict(
            X=X_train,
            y=y_train
        ),
        test=dict(
            X=X_test,
            y=y_test
        )
    )

    def process_results(results):
        if isinstance(results.probabilities, str) and results.probabilities == "predictions":
            target_values_enc = dataset.target.label_encoder.transform(dataset.target.values)
            results.probabilities = Encoder('one-hot', target=False, encoded_type=float).fit(target_values_enc).transform(results.predictions)
        is_numpy_like = hasattr(results.probabilities, "shape") and results.probabilities.shape
        if results.probabilities is None or is_numpy_like:
            return results
        raise ValueError(f"Unknown probabilities format: {results.probabilities}")

    return run_in_venv(__file__, "exec.py",
                       input_data=data, dataset=dataset, config=config,
                       process_results=process_results)

