# HOW-TO

## Run a benchmark
see [README](README.md#quickstart) for basic commands.

### Custom configuration

Default configuration properties are all defined and described in the `resources/config.yaml` file.

To override those configurations, create your custom `config.yaml` file under the `user_dir` . The application will automatically load this custom file and apply it on top of the defaults.

_Example of config.yaml:_
```yaml
---
input_dir: ~/data   # change the default input directory (where data files are loaded and/or downloaded).

frameworks:
  definition_file:  # this allows to add custom framework definitions (in {user}/frameworks.yaml) on top of the default ones.
    - '{root}/resources/frameworks.yaml'
    - '{user}/frameworks.yaml'

benchmarks:
  definition_dir:  # this allows to add custom benchmark definitions (under {user}/benchmarks) to the default ones.
    - '{user}/benchmarks'
    - '{root}/resources/benchmarks'
  constraints_file: # this allows to add custom constraint definitions (in {user}/constraints.yaml) on top of the default ones.
    - '{root}/resources/constraints.yaml'
    - '{user}/constraints.yaml'

aws:
  resource_files:  # this allows to automatically upload custom config + frameworks to the running instance (benchmark files are always uploaded).
    - '{user}/config.yaml'
    - '{user}/frameworks.yaml'
    - '{user}/extensions'

  use_docker: true  # you can decide to always use the prebuilt docker images on AWS.
```  

**Note:** configurations support the following placeholders:
- `{input}`: replaced by the value of config `input_dir`. Folder from which datasets are loaded (and/or downloaded) by default. Defaults to `~/.openml/cache`, but can also be overridden in a custom `config.yaml` or at the command line using `-i` or `--indir`.
- `{output}`: replaced by the value of config `output_dir`. Folder where all outputs (results, logs, predictions...) will be stored. Defaults to `./results`, but can also be overridden in a custom `config.yaml` or at the command line using `-o` or `--outdir`. 
- `{user}`: replaced by the value of config `user_dir`. Folder containing customizations (`config.yaml`, benchmark definitions, framework definitions...). Defaults to `~/.config/automlbenchmark`, but can be overridden at the command line using `-u` or `--userdir`.
- `{root}`: replaced by the value of config `root_dir`. The root folder of the `automlbenchmark` application: this is detected at runtime.

#### Run a framework with different (hyper-)parameters

Framework definitions accept a `params` dictionary for pass-through parameters, i.e. parameters that are directly accessible from the `exec.py` file in the framework integration executing the AutoML training.

_Example:_

In the definition below, the `n_estimators` and `verbose` params are passed directly to the `RandomForestClassifier`
```yaml
RandomForest:
  version: '0.21.3'
  project: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
  params:
    n_estimators: 2000
    verbose: true
```

**NOTE:** by convention, param names starting with `_` are filtered out (they are not passed to the classifier/regressor) but are used for custom logic in the `exec.py`.

_Example:_
 
In the definition below, the `_n_jobs` param is handled by custom code in `RandomForest/exec.py`: here it overrides the default `n_jobs` automatically calculated by the application (using all assigned cores).
```yaml
RandomForest:
  version: '0.21.3'
  project: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
  params:
    n_estimators: 2000
    _n_jobs: 1
```

## Add a benchmark

In this section, `benchmark` means a suite of datasets that can be used to feed any of the available frameworks, in combination with a set of constraints (time limit, cpus, memory) enforced by the application.

A benchmark definition will then consist in a [datasets definition](#datasets-definition) and a [constraints definition](#constraint-definition).

Each dataset must contain a training set and a test set. There can be multiple training/test splits, in which case each split is named a `fold`, so that the same dataset can be benchmarked multiple times using a different fold.

### Datasets definition

A dataset definition consists in a `yaml` file listing all the task/datasets that will used for the complete benchmark.

Default dataset definitions are available under folder `resources/benchmarks`.

Each task/dataset must have a `name` that should be unique (ignoring case) in the given definition file, it will also be used as an identifier, for example in the results.

This `name` can also be used on the command line (`-t` or `--task` argument) when we just want to execute a subset of the benchmark, often in combination with a specific fold (`-f` or `--fold` argument):
```bash
python runbenchmark.py randomforest validation -t bioresponse
python runbenchmark.py randomforest validation -t bioresponse eucalyptus
python runbenchmark.py randomforest validation -t bioresponse -f 0
python runbenchmark.py randomforest validation -t bioresponse eucalyptus -f 0 1 2
``` 

_Example:_

Following the [custom configuration](#custom-configuration), it is possible to override and/or add custom benchmark definitions by creating for example a `mybenchmark.yaml` file in your `user_dir/benchmarks`.

The benchmark can then be tested and then executed using the `1h4c` constraint:
```bash
python runbenchmark.py randomforest mybenchmark
python runbenchmark.py randomforest mybenchmark 1h4c
```


#### OpenML datasets

[OpenML] datasets are verified and annotated datasets, making them easy to consume.
However, the application doesn't directly consume those datasets today as the split between training data and test data is not immediately available.
For this we use [OpenML] tasks.

#### OpenML tasks

[OpenML] tasks provide ready to use datasets, usually split in 10-folds: for each fold, we have 1 training set and a test set.

The automlbenchmark application can directly consume those tasks using the following definition:
```yaml
- name: bioresponse
  openml_task_id: 9910
```
where `openml_task_id` allows accessing the OpenML task at `https://www.openml.org/t/{openml_task_id}` (in this example: <https://www.openml.org/t/9910>). 


#### OpenML studies

[OpenML] studies are a collection of OpenML tasks, for example <https://www.openml.org/s/218>.
The application doesn't directly support OpenML studies for now: they need to be converted into a proper benchmark definition file including all the tasks from the study, but we're thinking about improving this: cf. <https://github.com/openml/automlbenchmark/issues/61>.

#### File datasets

It is also possible to benchmark your own datasets, as soon as they follow some requirements:
- The data files should be in one of the currently supported format: [ARFF], [CSV] (ideally with header).
- Each dataset must contain at least one file for training data and one file for test data.
- If the dataset is represented as an archive (.zip, .tar, .tgz, .tbz) or a directory, then the data files must follow this naming convention to be detected correctly:
  - if there's only one file for training and one for test, they should be named `{name}_train.csv` and `{name}_test.csv` (in case of CSV files).
  - if there are multiple `folds`, they should follow a similar convention: `{name}_train_0.csv`, `{name}_test_0.csv``, {name}_train_1.csv`, `{name}_test_1.csv`, ...

_Example:_

Then the datasets can be declared in the benchmark definition file as follow:
```yaml
---

- name: example_csv
  dataset:
    train: /path/to/data/ExampleTraining.csv
    test:  /path/to/data/ExampleTest.csv
    target: TargetColumn
  folds: 1

- name: example_multi_folds
  dataset:
    train: 
      - /path/to/data/ExampleTraining_0.csv
      - /path/to/data/ExampleTraining_1.csv
    test:  
      - /path/to/data/ExampleTest_0.csv
      - /path/to/data/ExampleTest_1.csv
    target: TargetColumn
  folds: 2

- name: example_dir   # let's assume that the data folder contains 2 files: example_train.arff and example_test.arff
  dataset: 
    path: /path/to/data
    target: TargetColumn
  folds: 1

- name: example_dir_multi_folds   # let's assume that the data folder contains 6 files: example_train_0.arff, ..., example_train_2.arff, example_test_0.arff, ...
  dataset: 
    path: /path/to/data
    target: TargetColumn
  folds: 3

- name: example_archive  # let's assume that archive contains the same files as for example_dir_multi_folds
  dataset:
    path: /path/to/archive.zip
    target: TargetColumn
  folds: 3

- name: example_csv_http
  dataset:
    train: https://my.domain.org/data/ExampleTraining.csv
    test:  https://my.domain.org/data/ExampleTest.csv
    target: TargetColumn
  folds: 1

- name: example_archive_http
  dataset:
    path: https://my.domain.org/data/archive.tgz
    target: TargetColumn
  folds: 3

- name: example_autodetect
  dataset: /path/to/data/folder

- name: example_relative_to_input_dir
  dataset: "{input}/data/folder"

- name: example_relative_to_user_dir
  dataset:
    train: "{user}/data/train.csv"
    test: "{user}/data/test.csv"

```
**Note**:
- the naming convention is required only when `path` is pointing to a directory or an archive. If the files are listed explicitly, there's no constraint on the file names.
- the `target` attribute is optional but recommended, otherwise the application will try to autodetect the target:
  0. looking for a column named `target` or `class`.
  0. using the last column as a fallback.
- the `folds` attribute is also optional but recommended for those datasets as the default value is `folds=10` (default amount of folds in openml datasets), so if you don't have that many folds for your custom datasets, it is better to declare it explicitly here.
- Remote files are downloaded to the `input_dir` folder and archives are decompressed there as well, so you may want to change the value of this folder in your [custom config.yaml file](#custom-configuration) or specify it at the command line with the `-i` or `--indir` argument (by default, it points to the `~/.openml/cache` folder).

### Constraints definition

Now that we have defined a list of datasets, we also need to enforce some constraints on the autoML training.

Default constraint definitions are available in `resources/constraint.yaml`. When no constraint is specified at the command line, the `test` constraint definition is used by default.

THe application supports the following constraints:
- `folds` (default=10): tell the number of tasks that will be created by default for each dataset of the benchmark. For example, if all datasets support 10 folds, setting a constraint `folds: 2` will create a task only for the first 2 folds by default.
- `max_runtime_seconds` (default=3600): maximum time assigned for each individual benchmark task. This parameter is usually passed directly to the framework: if it doesn't respect the constraint, the application will abort the task after `2 * max_runtime_seconds`. In any case, the real task running time is always available in the results.
- `cores` (default=-1): amount of cores used for each automl task. If <= 0, it will try to use all cores.
- `max_mem_size_mb` (default=-1): amount of memory assigned to each automl task. If <= 0, then the amount of memory is computed from os available memory.
- `min_vol_size_mb` (default=-1): minimum amount of free space required on the volume. If <= 0, skips verification. If the requirement is not fulfilled, a warning message will be printed, but the task will still be attempted.

_Example:_

Following the [custom configuration](#custom-configuration), it is possible to override and/or add constraints by creating the following `constraints.yaml` file in your `user_dir`:

```yaml
---

test:
  folds: 1
  max_runtime_seconds: 120

1h16c:
  folds: 10
  max_runtime_seconds: 3600
  cores: 16

1h32c:
  folds: 10
  max_runtime_seconds: 3600
  cores: 32

4h16c:
  folds: 10
  max_runtime_seconds: 14400
  cores: 16
  min_vol_size_mb: 65536

8h16c:
  folds: 10
  max_runtime_seconds: 28800
  cores: 16
  min_vol_size_mb: 65536

```

The new constraints can now be passed on the command line when executing the benchmark:
```bash
python runbenchmark.py randomforest validation 1h16c
```

## Add an AutoML framework

Adding an AutoML framework consist in several steps:
 1. create a Python module that will contain everything related to the integration of this framework.
 1. define the framework in a [Framework definition](#framework-definition) file.
 1. write some integration code
   - to download/setup the framework dynamically: by convention, this is done by a `setup.sh` script defined in the module.
   - to run the framework using the data and constraints/parameters provided by the benchmark application: by convention, this is done by an `exec.py` script in the module, but it may require more files depending on the framework, for example if it runs on Python or R, Java...
   

### Framework definition

The framework definition consists in an entry in a `yaml` file with the framework name and some properties
 1. to describe the framework: `project`, `version`.
 1. to indicate the Python module with the integration code: `module` or `extends`.
 1. to pass optional parameters to the framework and/or the integration code: `params`.
 
Default framework definitions are defined in file `resources/frameworks.yaml`.

Following the [custom configuration](#custom-configuration), it is possible to override and/or add a framework definitions by creating a `frameworks.yaml` file in your `user_dir`.

See for example the `examples/custom/frameworks.yaml`:

```yaml
---

GradientBoosting:
  module: extensions.GradientBoosting
  version: '0.19.2'
  project: https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting
  params:
    n_estimators: 500

Stacking:
  module: extensions.Stacking
  version: '0.22.1'
  project: https://scikit-learn.org/stable/modules/ensemble.html#stacking
  params:
    _rf_params: {n_estimators: 200}
    _gbm_params: {n_estimators: 200}
    _linear_params: {penalty: elasticnet, loss: log}
#    _svc_params: {tol: 1e-3, max_iter: 1e5}
#    _final_params: {penalty: elasticnet, loss: log} # sgd linear
    _final_params: {max_iter: 1000}  # logistic/linear

H2OAutoML_nightly:
  module: frameworks.H2OAutoML
  setup_cmd: 'LATEST_H2O=`curl http://h2o-release.s3.amazonaws.com/h2o/master/latest` && pip install --no-cache-dir -U "http://h2o-release.s3.amazonaws.com/h2o/master/${{LATEST_H2O}}/Python/h2o-3.29.0.${{LATEST_H2O}}-py2.py3-none-any.whl"'
  version: 'nightly'

H2OAutoML_blending:
  extends: H2OAutoML
  params:
    nfolds: 0

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

By default, when generating a [docker image](README.md#in-docker-image), the image name is created as `automlbenchmark/{framework}:{version}-{branch}` with the framework name in lowercase, and `branch` being the branch of the `automlbenchmark` app (usually `stable`).
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
   - Most of them currently run using the same dependencies as the application, which is not recommended due to potential version conflicts (especially with `sklearn`). This was not a major constraint with the first frameworks implemented, but now, those integrations can and will be slightly changed to [run in their dedicated virtual environment](#frameworks-requiring-a-dedicated-virtual-env), using their own dependencies: cf. `RandomForest` and `examples/custom/extensions/Stacking` for examples.
- non-Python frameworks: those frameworks typically run in `R` or `Java` and don't provide any Python API. The integration is then still done by spawning the `R` or `Java` process from the `exec.py`: cf. `AutoWEKA`, `autoxgboost`.

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
from amlb.utils import as_cmd_args, call_script_in_same_dir, dir_of


def setup(*args, **kwargs):
    call_script_in_same_dir(__file__, "setup.sh", *args, **kwargs)


def run(*args, **kwargs):
    from .exec import run
    return run(*args, **kwargs)


def docker_commands(*args, **kwargs):
    return """
RUN {here}/setup.sh {args}
""".format(
        here=dir_of(__file__, True),
        args=' '.join(as_cmd_args(*args)),
    )

__all__ = (setup, run, docker_commands)

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
- the `run` function in `__init__py` prepares the data (in the application environment) before executing the `exec.py` in the dedicated `venv`. The call to `run_in_venv` is in charge of serializing the input, calling `exec.py` and deserializing + saving the results from `exec`.
- `exec.py`, when calls in the subprocess (function `__main__`), calls `call_run(run)` which deserializes the input (dataset + config) and passes it to the `run` function that just need to return a `result` object.

*Note*:

As the serialization/deserialization of `numpy` arrays can be costly for very large datasets, it is recommended to use dataset serialization only if the framework itself doesn't support loading datasets from files. 

This means that, in the `__init__.py` instead of implementing `run` as:
```python
X_train_enc, X_test_enc = impute(dataset.train.X_enc, dataset.test.X_enc)
data = dict(
    train=dict(
        X_enc=X_train_enc,
        y_enc=dataset.train.y_enc
    ),
    test=dict(
        X_enc=X_test_enc,
        y_enc=dataset.test.y_enc
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
- the framework is then executed by building a command manually in `exec.py`, runming it in a subprocess, and finally collecting the results generated by the subprocess. For example, in `autoxgboost/exec.py`:
  ```python
  with Timer() as training:
      run_cmd(r"""Rscript --vanilla -e "source('{script}'); run('{train}', '{test}', target.index = {target_index}, '{output}', {cores}, time.budget = {time_budget})" """.format(
          script=os.path.join(here, 'exec.R'),
          train=dataset.train.path,
          test=dataset.test.path,
          target_index=dataset.target.index+1,
          output=config.output_predictions_file,
          cores=config.cores,
          time_budget=config.max_runtime_seconds
      ), _live_output_=True)
  ```
  Here, the `exec.R` script is also responsible to save the predictions in the expected format.

#### Add a default framework

Using the instructions above:
 1. verify that there is an issue created under <https://github.com/openml/automlbenchmark/issues> for the framework you want to add, or create one.
 1. create a private branch for your integration changes.
 1. create the framework module (e.g. `MyFramework`) under `frameworks` folder.
 1. define the module (if possible without any `params`) in `resources/frameworks.yaml`.
 1. try to setup the framework: 
    > python runbenchmark.py myframework -s only
 1. fixes the framework setup until it works: the setup being usually a simple `setup.sh` script, you should be able to test it directly without using the application.
 1. try to run simple test against one fold using defaults (`test` benchmark and `test` constraints):
    > python runbenchmark.py myframework -f 0
 1. fix the module integration code until the test produce all results with no error (if the integration generated an error, it is visible in the results).
 1. if this works, validate it against the `validation` dataset using one fold:
    > python runbenchmark.py myframework validation 1h4c -f 0
 1. if this works, try to run it in docker to validate the docker image setup: 
    > python runbenchmark.py myframework -m docker
 1. if this works, try to run it in aws: 
    > python runbenchmark.py myframework -m aws
 1. create a pull request, and ask a review from authors of `automlbenchmark`: they'll also be happy to help you during this integration.

#### Add a custom framework

You may want to integrate a framework without wanting to make this publicly available.

In this case, as we've seen above, there's always the possibility to integrate your framework in a custom `user_dir`.

Using the instructions above:
 1. define what is (or will be) your custom `user_dir` for this framework.
 1. ensure it contains a `config.yaml`, otherwise create one (for example copy [this one](#custom-configuration) or `examples/custom/config.yaml`).
 1. create the framework module somewhere under this `user_dir`, e.g. `{user_dir}/extensions/MyFramework`.
 1. define the module in `{user_dir}/frameworks.yaml` (create the file if needed).
 1. follow the same steps as for a "default" framework to implement the integration: setup, test, ... except that you always need to specify the `user_dir`, e.g. for testing:
    > python runbenchmark.py myframework -f 0 -u {user_dir}
 1. there may be some issues when trying to build the docker image when the framework is in a custom folder, as all the files should be under the docker build context: solving this probably requires a multi-stage build, needs more investigation. For now, if you really need a docker image, you can either build it manually, or simply copy the `extensions` folder temporarily under `automlbenchmark`.
 1. even without docker image, you can run the framework on AWS, as soon as the custom `config.yaml`, `frameworks.yaml` and `extensions` folder are made available as AWS resources: cf. again the [custom configuration](#custom-configuration). The application will copy those files to the EC2 instances into a local `user_dir` and will be able to setup the framework there.


## Troubleshooting guide

### Where are the results?

By default, the results for a benchmark execution are made available in a subfolder under `output_dir` (if not specified by `-o my_results`, then this is under `{cwd}/results`).

This subfolder is named `{framework}_{benchmark}_{constraint}_{mode}_{timestamp}`.

So that for example:
```bash
python runbenchmark.py randomforest
```
will create a subfolder `randomforest_test_test_local_20200108T184305`,

and:
```bash
python runbenchmark.py randomforest validation 1h4c -m aws
```
will create a subfolder `randomforest_validation_1h4c_aws_20200108T184305`.


Then each subfolder contains:
 - a `score` folder with a `results.csv` file concatenating the results from all the tasks in the benchmark, as well as potentially other individual results for each task.  
 - a `predictions` folder with the predictions for each task in the benchmark.
 - a `logs` folder: only if benchmark was executed with `-o output_dir` argument.
 - possibly more folders if the framework saves additional artifacts.
 
 Also the `output_dir` contains a `results.csv` concatenating **ALL results** from all subfolders.


### Where are the logs?

By default the application logs are available under `{cwd}/logs` if the benchmark is executed without specifying the `output_dir`, otherwise, they'll be available under the `logs` subfolder in the benchmark results (see [Where are the results?](#where-are-the-results)).

The application can collect various logs:
- local benchmark application logs: those are always collected. For each run, the application generated 2 log files locally:
  - `runbenchmark_{timestamp}.log`: contains logs for the application only (from DEBUG level).
  - `runbenchmark_{timestamp}_full.log`: contains logs for the application + other Python libraries (from INFO level); e.g. `boto3` logs when running in `aws` mode.
- remote application logs: for `aws` mode only, logs generated on the remote instances are automatically downloaded to the results folder, together with other result artifacts. 
- framework logs (optional): if the framework integration supports it, it is possible to ask for the framework logs by creating a custom framework definition as follow:
  ```yaml
  H2OAutoML:
    extends: H2OAutoML
    params:
      _save_artifacts: ['logs']
  ```

### Profiling the application

Currently, the application provides a global flag `--profiling` to activate profiling for some specific methods that can be slow or memory intensive:
```bash
python runbenchmark.py randomforest --profiling
```
All methods/functions are not profiled, but if you need to profile more, you just need to decorate the function with the `@profile()` decorator (from `amlb.utils`).

#### Memory usage
_Examples of memory info when using this custom profiling_:
```text
[PROFILING] `amlb.datasets.openml.OpenmlDatasplit.data` returned object size: 45.756 MB.
[PROFILING] `amlb.datasets.openml.OpenmlDatasplit.data` memory change; process: +241.09 MB/379.51 MB, resident: +241.09 MB/418.00 MB, virtual: +230.01 MB/4918.16 MB.
...
[PROFILING] `amlb.data.Datasplit.release` executed in 0.007s.
[PROFILING] `amlb.data.Datasplit.release` memory change; process: -45.73 MB/238.80 MB, resident: +0.00 MB/414.60 MB, virtual: +0.00 MB/4914.25 MB.
```

#### Methods duration
_Examples of method duration info when using this custom profiling_:
```text
[PROFILING] `amlb.datasets.openml.OpenmlLoader.load` executed in 7.456s.
...
[PROFILING] `amlb.data.Datasplit.X_enc` executed in 6.570s.
```

### Python library version conflict 
see [Framework integration](#frameworks-requiring-a-dedicated-virtual-env)

### Framework setup is not executed

Try the following:
- force the setup using the `-s only` or `-s force` arg on the command line:
  - `-s only` or `--setup=only` will force the setup and skip the benchmark run.
  - `-s force` or `--setup=force` will force the setup and run the benchmark immediately.
- delete the `.marker_setup_safe_to_delete` from the framework module and try to run the benchmark again. This marker file is automatically created after a successful setup to avoid having to execute it each tine (setup phase can be time-consuming), this marker then prevents auto-setup, except if the `-s only` or `-s force` args above are used.

[README]: ./README.md
[OpenML]: https://openml.org
[ARFF]: https://waikato.github.io/weka-wiki/formats_and_processing/arff_stable/
[CSV]: https://tools.ietf.org/html/rfc4180
[Docker]: https://docs.docker.com/
