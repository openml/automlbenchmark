from amlb.resources import config as rconfig
from amlb.utils import call_script_in_same_dir

def setup(*args, **kwargs):
    call_script_in_same_dir(__file__, "setup.sh", rconfig().root_dir, *args, **kwargs)


def run(dataset, config):
    from frameworks.shared.caller import run_in_venv
    enc = config.framework_params.get('_enc', False)
    if enc:
        from amlb.datautils import impute
        X_train_enc, X_test_enc = impute(dataset.train.X_enc, dataset.test.X_enc)
        data = dict(
            train=dict(
                X_enc=X_train_enc,
                y_enc=dataset.train.y_enc
            ),

            test=dict(
                X_enc=X_test_enc,
                y_enc=dataset.test.y_enc
            ),

            problem_type=dataset.type.name  
        )
    else:
        data = dict(
            train=dict(data=dataset.train.data),
            test=dict(data=dataset.test.data),
            target=dict(
                name=dataset.target.name,
                classes=dataset.target.values
            ),
            columns=[(f.name, ('object' if f.is_categorical(strict=False)  # keep as object everything that is not numerical
                            else 'int' if f.data_type == 'integer'
                            else 'float')) for f in dataset.features],
            problem_type=dataset.type.name 
        )

    return run_in_venv(__file__, "exec.py",
        input_data=data, dataset=dataset, config=config)