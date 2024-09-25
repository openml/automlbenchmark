# AutoGluon

To run v0.8.2: ```python3 ../automlbenchmark/runbenchmark.py autogluon ...```

To run mainline: ```python3 ../automlbenchmark/runbenchmark.py autogluon:latest ...```

## Callbacks
Callbacks from the `autogluon.core.callbacks` module can be configured from the configuration file by specifying the classname and any hyperparameters (see AutoGluonES example below).
This configuration is case-sensitive.
```yaml
AutoGluonES:
  extends: AutoGluon
  params:
    callbacks:
      EarlyStoppingEnsembleCallback:
        patience: 5
```

## Running with Learning Curves Enabled

To run with learning curves enabled, you must define a new framework in the frameworks YAML file which enables the appropriate parameters. Instructions on how to do this are listed here:
https://openml.github.io/automlbenchmark/docs/extending/framework/

To summarize these steps:
1. Add one of the below frameworks (or your own) to [this](https://github.com/openml/automlbenchmark/blob/master/examples/custom/frameworks.yaml) frameworks.yaml file. Alternatively, you can create your own custom user_dir.
3. Navigate to the root directory of the automlbenchmark repository
4. Run `python3 runbenchmark.py {YOUR_CUSTOM_FRAMEWORK_NAME} -u {PATH_TO_USER_DIR} ...` \
    where PATH_TO_USER_DIR is the path to the user_dir containing your frameworks.yaml file. If you used the example user_dir, replace the path with `examples/custom`
5. For example, if running the most basic framework listed below in the example user_dir, the command would look like: \
    `python3 runbenchmark.py AutoGluon_curves_true -u examples/custom ...`

	
### Sample Framework Definitions
```yaml
# simplest usage
AutoGluon_curves_true:
  extends: AutoGluon
  params:
    learning_curves: True
    _save_artifacts: ['learning_curves']
```
```yaml
# including test data
AutoGluon_curves_test:
  extends: AutoGluon
  params:
    learning_curves: True
    _include_test_during_fit: True
    _save_artifacts: ['learning_curves']
```
```yaml
# parameterizing learning_curves dictionary
AutoGluon_curves_parameters:
  extends: AutoGluon
  params:
    learning_curves:
      use_error: True
    _curve_metrics:
        binary: ["log_loss", "accuracy", "precision"]
        regression: ['root_mean_squared_error', 'median_absolute_error', 'r2']
        multiclass: ["accuracy", "precision_weighted", "recall_weighted"]
    _include_test_during_fit: True
    _save_artifacts: ['learning_curves']
```
```yaml
# adding custom hyperparameters
Defaults: &defaults
  NN_TORCH:
    - num_epochs: 1000
      epochs_wo_improve: 999999999
  GBM:
    - num_boost_round: 20000
      ag_args_fit:
        early_stop: 999999999
  XGB:
    - n_estimators: 20000
      ag_args_fit:
        early_stop: 999999999

AutoGluon_curves_hyperparameters:
  extends: AutoGluon
  params:
    hyperparameters:
      <<: *defaults
    learning_curves:
      use_error: True
    _curve_metrics:
        binary: ["log_loss", "accuracy", "precision"]
        regression: ['root_mean_squared_error', 'median_absolute_error', 'r2']
        multiclass: ["accuracy", "precision_weighted", "recall_weighted"]
    _include_test_during_fit: True
    _save_artifacts: ['learning_curves']
```
