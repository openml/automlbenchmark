# Adding an AutoML Framework

!!! warning "Rewrite in progress"

    Most information on this page is accurate, and it should be complete enough to use.
    However, it hasn't been updated to make use of `mkdocs-materials` features, and
    _might_ have some outdated examples. Contributions welcome.

## Add an AutoML framework

Adding an AutoML framework consist in several steps:

 1. create a Python module that will contain everything related to the integration of this framework.
 1. define the framework in a [Framework definition](#framework-definition) file.
 1. write some integration code
   - to download/setup the framework dynamically: by convention, this is done by a `setup.sh` script defined in the module.
   - to run the framework using the data and constraints/parameters provided by the benchmark application: by convention, this is done by an `exec.py` script in the module, but it may require more files depending on the framework, for example if it runs on Python or R, Java...
   

### Framework definition

The framework definition consists in an entry in a `yaml` file with the framework name and some properties

 1. to describe the framework and define which version will be used: `project`, `version`.
 1. to indicate the Python module with the integration code: `module` or `extends`.
 1. to pass optional parameters to the framework and/or the integration code: `params`.
 
Default framework definitions are defined in file `resources/frameworks.yaml` in lexicographic order, 
where `version` should be set to `stable`, which will point dynamically to the most recent official release available.

Frameworks that offer the possibility to test cutting edge version (e.g. nightly builds, 
`dev`/`master` repo, ...) can add an entry to `resources/frameworks_latest.yaml`, where `version` should be set to `latest`.

Maintainers of this repository try to regularly — ideally, every quarter — create a 
framework definition using frozen framework versions in order to favour the reproducibility of the published benchmarks.

Following the [custom configuration](../using/configuration.md#custom-configurations), 
it is possible to override and/or add a framework definitions by creating a `frameworks.yaml` file in your `user_dir`.

See for example the `examples/custom/frameworks.yaml`:

```yaml
---

GradientBoosting:
  module: extensions.GradientBoosting
  project: https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting
  params:
    n_estimators: 500

Stacking:
  module: extensions.Stacking
  project: https://scikit-learn.org/stable/modules/ensemble.html#stacking
  params:
    _rf_params: {n_estimators: 200}
    _gbm_params: {n_estimators: 200}
    _linear_params: {penalty: elasticnet, loss: log}
#    _svc_params: {tol: 1e-3, max_iter: 1e5}
#    _final_params: {penalty: elasticnet, loss: log} # sgd linear
    _final_params: {max_iter: 1000}  # logistic/linear

autosklearn_latest:
  extends: autosklearn
  version: latest
  description: "this will use master branch from the autosklearn repository instead of the fixed version"

autosklearn_mybranch:
  extends: autosklearn
  version: mybranch
  description: "this will use mybranch branch from the autosklearn repository instead of the fixed version"

autosklearn_oldgen:
  extends: autosklearn
  version: "0.7.1"
  description: "this will use the latest autosklearn version from the old generation"

H2OAutoML_nightly:
  module: frameworks.H2OAutoML
  setup_cmd: 'LATEST_H2O=`curl http://h2o-release.s3.amazonaws.com/h2o/master/latest` && pip install --no-cache-dir -U "http://h2o-release.s3.amazonaws.com/h2o/master/${{LATEST_H2O}}/Python/h2o-3.29.0.${{LATEST_H2O}}-py2.py3-none-any.whl"'
  version: 'nightly'

H2OAutoML_custom:
  extends: H2OAutoML
  params:
    nfolds: 3
    stopping_tolerance: 0.05
```

This example shows

- the definitions for 2 new frameworks: `GradientBoosting` and `Stacking`. 
  - Those definitions (optionally) externalize some parameters (e.g. `n_estimators`): the `params` property always appears in json format in the results, so that we can clearly see what has been tuned when analyzing the results later.
  - Note that the module is case sensitive and should point to the module containing the integration code.
  - The application will search for modules from the sys path, which includes the application `root_dir` and the `user_dir`: 
    - that's why the default frameworks use `module: frameworks.autosklearn` for example, 
    - and the example above can use `module: extensions.GradientBoosting` because those examples must be run by setting the `user_dir` to `examples/config`, e.g. 
      > `python runbenchmark.py gradientboosting -u examples/custom`.
- a custom definition (`H2OAutoML_nightly`) for the existing `frameworks.H2OAutoML` module, allowing to reuse the module for a dynamic version of the module:
    - the `setup_cmd` is executed after the default setup of the module, so it can be used to make additional setup. To customize the setup, it is possible to use:
      - `setup_args: my_version` (only if the `setup.sh` in the framework module supports new arguments).
      - `setup_cmd` (as shown in this example). 
      - `setup_script: my_additional_setup.sh`.
- 2 custom definitions (`H2OAutoML_blending` and `H2OAutoML_custom`) simply extending the existing `H2OAutoML` definition (therefore inheriting from all its properties, including the `module` one), but overriding the `params` property, thus allowing to provide multiple "flavours" of the same framework.  

The frameworks defined in this example can then be used like any other framework as soon as both the framework module and the definition file are made available to the application: in our case, this is done by the creation of the integration modules under `examples/custom/extensions` and by exposing the definitions in `examples/custom/frameworks.yaml` thanks to the entry in `examples/custom/config.yaml`:
```yaml
frameworks:
  definition_file:  # this allows to add custom framework definitions (in {user}/frameworks.yaml) on top of the default ones.
    - '{root}/resources/frameworks.yaml'
    - '{user}/frameworks.yaml'
```

By pointing the `user_dir` to `examples/custom`, our `config.yaml` is also loaded, and we can use the new frameworks:
```bash
python runbenchmark.py gradientboosting -u examples/custom
python runbenchmark.py stacking -u examples/custom
python runbenchmark.py h2oautoml_blending -u examples/custom
```

*Note:*

By default, when generating a docker image, the image name is created as `automlbenchmark/{framework}:{version}-{branch}` with the framework name in lowercase, and `branch` being the branch of the `automlbenchmark` app (usually `stable`).
However, it is possible to customize this image name as follow:
```yaml
MyFramework:
  version: 1.0
  module: extensions.MyFramework
  docker:
    author: my_docker_repo
    image: my_image
    tag: my_tag
```
which will result in the docker image name `my_docker_repo/my_image:my_tag-{branch}`, with `branch` still being the branch of the application.


### Framework integration

If the framework definition allows to use the new framework from the application, the (not so) hard part is to integrate it.

There are already several frameworks already integrated under `frameworks` directory (+ the examples under `examples/custom`), so the best starting point when adding a new framework is to first look at the existing ones.

Among the existing frameworks, we can see different type of integrations:

- trivial integration: these are frameworks running on Python and using dependencies (`numpy`, `sklearn`) already required by the application itself. These are not really AutoML toolkits, but rather integrations using `sklearn` to provide a reference when analyzing the results: cf. `constantpredictor`, `DecisionTree`.
- Python API integration: these are frameworks that can be run directly from Python: cf. `autosklearn`, `H2OAutoML`, `TPOT`, `RandomForest`, `TunedRandomForest`.
   - contrary to the trivial integration, those require a `setup` phase.
   - Most of them currently run using the same dependencies as the application, which is not recommended due to potential version conflicts (especially with `sklearn`). This was not a major constraint with the first frameworks implemented, but now, those integrations can and will be slightly changed to [run in their dedicated virtual environment], using their own dependencies: cf. `RandomForest` and `examples/custom/extensions/Stacking` for examples.
- non-Python frameworks: those frameworks typically run in `R` or `Java` and don't provide any Python API. The integration is then still done by spawning the `Java` or `R` process from the `exec.py`: cf. `AutoWEKA` or `ranger`, respectively.

#### Recommended structure

By convention, the integration is done using the following structure:

```text
frameworks/autosklearn/
|-- __init__.py
|-- exec.py
|-- requirements.txt
`-- setup.sh
```

Please note however, that this structure is not a requirement, the only requirement is the contract exposed by the integration module itself, i.e. by the `__init__.py` file.

A simple `__init__.py` would look like this:

```python
from amlb.utils import call_script_in_same_dir


def setup(*args, **kwargs):
    call_script_in_same_dir(__file__, "setup.sh", *args, **kwargs)


def run(*args, **kwargs):
    from .exec import run
    return run(*args, **kwargs)

```

where we see that the module should expose (only `run` is actually required) the following functions:

- `setup` (optional): called by the application to setup the given framework, usually by simply running a `setup.sh` script that will be responsible for potentially creating a local virtual env, downloading and installing the dependencies. 
   The `setup` function can also receive the optional `setup_args` param from the [framework definition](#framework-definition) as an argument. 
- `run`: called by the benchmark application to execute a task against the framework, using the selected dataset and constraints. We will describe the parameters in detail below, for now, just note that by convention, we just load the `exec.py` file from the module and delegate the execution to its `run` function.
- `docker_commands` (optional): called by the application to collect docker instructions that are specific to the framework. If the framework requires a `setup` phase, then the string returned by this function should at least ensure that the setup is also executed during the docker image creation, that's one reason why it is preferable to do all the setup in a `setup.sh` script, to allow the docker support above.

#### Frameworks with Python API

##### Frameworks requiring a dedicated virtual env

For frameworks with Python API, we may worry about version conflicts between the packages used by the application (e.g. `sklearn`, `numpy`, `pandas`) and the ones required by the framework.

In this case, the integration is slightly different as you can see with the `RandomForest` integration allowing to use any version of `sklearn`.

This is the basic structure after the creation of the dedicated Python virtual environment during setup:
```text
frameworks/RandomForest/
|-- __init__.py
|-- exec.py
|-- requirements.txt
|-- setup.sh
`-- venv/
    `-- (this local virtual env is created by the frameworks/shared/setup.sh)
```

Noticeable differences with a basic integration:

- the `venv` is created in `setup.sh` by passing the current dir when sourcing the `shared/setup.sh` script: `. $HERE/../shared/setup.sh $HERE`.
- the `run` function in `__init__.py` prepares the data (in the application environment) before executing the `exec.py` in the dedicated `venv`. The call to `run_in_venv` is in charge of serializing the input, calling `exec.py` and deserializing + saving the results from `exec`.
- `exec.py`, when calls in the subprocess (function `__main__`), calls `call_run(run)` which deserializes the input (dataset + config) and passes it to the `run` function that just need to return a `result` object.

*Note A*:

As the serialization/deserialization of `numpy` arrays can be costly for very large datasets, it is recommended to use dataset serialization only if the framework itself doesn't support loading datasets from files. 

This means that, in the `__init__.py` instead of implementing `run` as:
```python
data = dict(
    train=dict(
        X=dataset.train.X,
        y=dataset.train.y
    ),
    test=dict(
        X=dataset.test.X,
        y=dataset.test.y
    )
)

return run_in_venv(__file__, "exec.py",
                   input_data=data, dataset=dataset, config=config)
```
it could simply expose the dataset paths (the application avoids loading the data if not explicitly needed by the framework):
```python
data = dict(
    target=dict(name=dataset.target.name),
    train=dict(path=dataset.train.path),
    test=dict(path=dataset.test.path)
)
return run_in_venv(__file__, "exec.py",
                   input_data=data, dataset=dataset, config=config)
```

*Note B*:

The serialization/deserialization of data between the main process and the framework process can be customized using the `options` parameter:
The allowed options for (de)serialization are defined by the object `amlb.utils.serialization.ser_config`.

For example:
```python
data = dict(
    train=dict(
        X=dataset.train.X,
        y=dataset.train.y
    ),
    test=dict(
        X=dataset.test.X,
        y=dataset.test.y
    )
)

options = dict(
    serialization=dict(sparse_dataframe_deserialized_format='dense')
)
return run_in_venv(__file__, "exec.py",
                   input_data=data, dataset=dataset, config=config, options=options)
```



#### Other Frameworks

Integration of frameworks without any Python API is done in similar way, for example:

```text
frameworks/AutoWEKA/
|-- __init__.py
|-- exec.py
|-- requirements.txt
|-- setup.sh
`-- lib/
    `-- (this is where the framework dependencies go, usually created by setup.sh)
```
or
```text
frameworks/ranger/
|-- __init__.py
|-- exec.R
|-- exec.py
|-- requirements.txt
`-- setup.sh
```

Here are the main differences:
- the `setup` phase is identical, but if at runtime, some executable file or library is required that need to be installed locally (as opposed to globally: for example, `R` or `java` executable are usually installed globally), we just recommend to put everything under the integration module (for example in `lib` and/or `bin` subfolders as for `AutoWEKA`). This is also true for some Python frameworks (cf. `hyperoptsklearn` integration for example, where the modules are loaded from `frameworks/hyperoptsklearn/lib/hyperopt-sklearn`).
- the framework is then executed by building a command manually in `exec.py`, running it in a subprocess, and finally collecting the results generated by the subprocess. For example, in `ranger/exec.py`:
  ```python
  with Timer() as training:
    run_cmd(("Rscript --vanilla -e \""
             "source('{script}'); "
             "run('{train}', '{test}', '{output}', cores={cores}, meta_results_file='{meta_results}', task_type='{task_type}')"
             "\"").format(
        script=os.path.join(here, 'exec.R'),
        train=dataset.train.path,
        test=dataset.test.path,
        output=config.output_predictions_file,
        meta_results=meta_results_file,
        task_type=config.type,
        cores=config.cores
    ), _live_output_=True)
  ```
  Here, the `exec.R` script is also responsible to save the predictions in the expected format.


#### Add a default framework

Is called "default framework" an AutoML framework whose integration is available on `master` branch under the `frameworks` folder, and with a simple definition in `resources/frameworks.yaml`.  

*NOTE:*
There are a few requirements when integrating a new default framework:

- The code snippet triggering the training should use only defaults (no AutoML hyper parameters), plus possibly a generic `**kwargs` in order to support `params` section in custom framework definitions.  In other words, one of the requirements for being included in the benchmark is that the framework is submitted without any tweaks to default settings.  This is to prevent submissions (systems) from overfitting or tuning to the benchmark.
- There must be a way to limit the runtime of the algorithm (a maximum runtime parameter).
- Exceptions:
  - the problem type ("classification", "regression", "binary", "multiclass"): this is available through `config.type` or `dataset.type`. 
  - information about data, for example the column types: available through the `dataset` object.
  - time, cpu and memory constraints: those must be provided by the benchmark application through the `config` object.  
  - the objective function: provided by `config.metric` (usually requires a translation for a given framework).
  - seed: provided by `config.seed`
  - paths to folders (output, temporary...): if possible, use `config.output_dir` or a subfolder (see existing integrations).
- The default framework definition in `resources/frameworks.yaml` shouldn't have any `params` section: this `params` section is intended for custom definitions, not default ones.
```yaml
good_framework:
   version: "0.0.1"
   project: "http://go.to/good_framework"

bad_framework:
   version: "0.0.1"
   project: "http://go.to/bad_framework"
   params: 
     enable_this: true
     use: ['this', 'that']
```

Using the instructions above:

 1. verify that there is an issue created under <https://github.com/openml/automlbenchmark/issues> for the framework you want to add, or create one.
 1. create a private branch for your integration changes.
 1. create the framework module (e.g. `MyFramework`) under `frameworks` folder.
 1. define the module (if possible without any `params`) in `resources/frameworks.yaml`.
 1. try to setup the framework: 
    > python runbenchmark.py myframework -s only
 1. fixes the framework setup until it works: the setup being usually a simple `setup.sh` script, you should be able to test it directly without using the application.
 1. try to run simple test against one fold using defaults (`test` benchmark and `test` constraints) with the `-Xtest_mode` that will trigger additional validations:
    > python runbenchmark.py myframework -f 0 -Xtest_mode
 1. fix the module integration code until the test produce all results with no error (if the integration generated an error, it is visible in the results).
 1. if this works, validate it against the `validation` dataset using one fold:
    > python runbenchmark.py myframework validation 1h4c -f 0 -Xtest_mode
 1. if this works, try to run it in docker to validate the docker image setup: 
    > python runbenchmark.py myframework -m docker
 1. if this works, try to run it in aws: 
    > python runbenchmark.py myframework -m aws
 1. add a brief description of the framework to the documentation in [docs/website/framework.html](GITHUB/docs/website/frameworks.html) following the same formatting as the other entries.
 1. create a pull request, and ask a review from authors of `automlbenchmark`: they'll also be happy to help you during this integration.

#### Add a custom framework

You may want to integrate a framework without wanting to make this publicly available.

In this case, as we've seen above, there's always the possibility to integrate your framework in a custom `user_dir`.

Using the instructions above:

 1. define what is (or will be) your custom `user_dir` for this framework.
 1. ensure it contains a `config.yaml`, otherwise create one (for example copy [this one](../using/configuration.md#custom-configurations) or `examples/custom/config.yaml`).
 1. create the framework module somewhere under this `user_dir`, e.g. `{user_dir}/extensions/MyFramework`.
 1. define the module in `{user_dir}/frameworks.yaml` (create the file if needed).
 1. follow the same steps as for a "default" framework to implement the integration: setup, test, ... except that you always need to specify the `user_dir`, e.g. for testing:
    > python runbenchmark.py myframework -f 0 -u {user_dir}
 1. there may be some issues when trying to build the docker image when the framework is in a custom folder, as all the files should be under the docker build context: solving this probably requires a multi-stage build, needs more investigation. For now, if you really need a docker image, you can either build it manually, or simply copy the `extensions` folder temporarily under `automlbenchmark`.
 1. even without docker image, you can run the framework on AWS, as soon as the custom `config.yaml`, `frameworks.yaml` and `extensions` folder are made available as AWS resources: cf. again the [custom configuration](../using/configuration.md#custom-configurations). The application will copy those files to the EC2 instances into a local `user_dir` and will be able to setup the framework there.


## Using a Different Hyperparameter Configuration

When you want to use an existing framework integration with a different hyperparameter
configuration, it is often enough to write only a custom framework definition without
further changes. 

Framework definitions accept a `params` dictionary for pass-through parameters, 
i.e., parameters that are directly accessible from the `exec.py` file in the framework 
integration executing the AutoML training. *Most* integration scripts use this to
overwrite any (default) hyperparameter value. Use the `extends` field to indicate
which framework definition to copy default values from, and then add any fields to
overwrite. In the example below the `n_estimators` and `verbose` params are passed 
directly to the `RandomForestClassifier`, which will now train only 200 trees
(default is 2000):

```yaml
RandomForest_custom:
  extends: RandomForest
  params:
    n_estimators: 200
    verbose: true
```

This new definition can be used as normal: 
```
python runbenchmark.py randomforest_custom ...
```

!!! note
    By convention, param names starting with `_` are filtered out (they are not passed 
    to the framework) but are used for custom logic in the `exec.py`. For example, the
    `_save_artifact` field is often used to allow additional artifacts, such as logs or
    models, to be saved.
