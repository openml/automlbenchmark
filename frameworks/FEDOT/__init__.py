from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.datautils import Encoder
from amlb.resources import config as rconfig
from amlb.utils import call_script_in_same_dir


def setup(*args, **kwargs):
    call_script_in_same_dir(__file__, "setup.sh", *args, **kwargs)


def run(dataset: Dataset, config: TaskConfig):
    from frameworks.shared.caller import run_in_venv

    data = dict(
        train=dict(
            X_enc=dataset.train.X_enc,
            y_enc=dataset.train.y_enc
        ),
        test=dict(
            X_enc=dataset.test.X_enc,
            y_enc=dataset.test.y_enc
        )
    )

    def process_results(results):
        if results.probabilities is not None and not results.probabilities.shape:  # numpy load always return an array
            prob_format = results.probabilities.item()
            if prob_format == "predictions":
                target_values_enc = dataset.target.label_encoder.transform(dataset.target.values)
                results.probabilities = Encoder('one-hot', target=False, encoded_type=float).fit(
                    target_values_enc).transform(results.predictions)
            else:
                raise ValueError(f"Unknown probabilities format: {prob_format}")
        return results

    return run_in_venv(__file__, "exec.py",
                       input_data=data, dataset=dataset, config=config,
                       process_results=process_results)
