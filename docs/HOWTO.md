# HOW-TO

  * [Run a benchmark](#run-a-benchmark)
     * [Custom configuration](#custom-configuration)
        * [Run a framework with different (hyper-)parameters](#run-a-framework-with-different-hyper-parameters)
     * [Advanced AWS Support](#advanced-aws-support) 
  * [Add a benchmark](#add-a-benchmark)
     * [Datasets definition](#datasets-definition)
        * [OpenML datasets](#openml-datasets)
        * [OpenML tasks](#openml-tasks)
        * [OpenML studies](#openml-studies)
        * [File datasets](#file-datasets)
     * [Constraints definition](#constraints-definition)
  * [Add an AutoML framework](#add-an-automl-framework)
     * [Framework definition](#framework-definition)
     * [Framework integration](#framework-integration)
        * [Recommended structure](#recommended-structure)
        * [Frameworks with Python API](#frameworks-with-python-api)
           * [Frameworks requiring a dedicated virtual env](#frameworks-requiring-a-dedicated-virtual-env)
        * [Other Frameworks](#other-frameworks)
        * [Add a default framework](#add-a-default-framework)
        * [Add a custom framework](#add-a-custom-framework)
  * [Analyze the results](#analyze-the-results)
     * [Results file format](#results-file-format)
     * [Predictions](#predictions)
     * [Extract more information](#extract-more-information)
  * [Troubleshooting guide](#troubleshooting-guide)
     * [Where are the results?](#where-are-the-results)
     * [Where are the logs?](#where-are-the-logs)
     * [Profiling the application](#profiling-the-application)
        * [Memory usage](#memory-usage)
        * [Methods duration](#methods-duration)
     * [Python library version conflict](#python-library-version-conflict)
     * [Framework setup is not executed](#framework-setup-is-not-executed)

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
    - '{user}/constraints.yaml'
    - '{user}/extensions'

  use_docker: true  # you can decide to always use the prebuilt docker images on AWS.
```  

**Note:** configurations support the following placeholders:
- `{input}`: replaced by the value of config `input_dir`. Folder from which datasets are loaded (and/or downloaded) by default. Defaults to `~/.openml/cache`, but can also be overridden in a custom `config.yaml` or at the command line using `-i` or `--indir`.
- `{output}`: replaced by the value of config `output_dir`. Folder where all outputs (results, logs, predictions...) will be stored. Defaults to `./results`, but can also be overridden in a custom `config.yaml` or at the command line using `-o` or `--outdir`. 
- `{user}`: replaced by the value of config `user_dir`. Folder containing customizations (`config.yaml`, benchmark definitions, framework definitions...). Defaults to `~/.config/automlbenchmark`, but can be overridden at the command line using `-u` or `--userdir`.
- `{root}`: replaced by the value of config `root_dir`. The root folder of the `automlbenchmark` application: this is detected at runtime.

**Note:** It is possible to have multiple configuration files: just create a folder for each `config.yaml` file and use that folder as your `user_dir` using `-u /path/to/config/folder`


#### Run a framework with different (hyper-)parameters

Framework definitions accept a `params` dictionary for pass-through parameters, i.e. parameters that are directly accessible from the `exec.py` file in the framework integration executing the AutoML training.

_Example:_

In the definition below, the `n_estimators` and `verbose` params are passed directly to the `RandomForestClassifier`
```yaml
RandomForest_custom:
  extends: RandomForest
  params:
    n_estimators: 2000
    verbose: true
```

**NOTE:** by convention, param names starting with `_` are filtered out (they are not passed to the classifier/regressor) but are used for custom logic in the `exec.py`.

_Example:_
 
In the definition below, the `_n_jobs` param is handled by custom code in `RandomForest/exec.py`: here it overrides the default `n_jobs` automatically calculated by the application (using all assigned cores).
```yaml
RandomForest_custom:
  extends: RandomForest
  params:
    n_estimators: 2000
    _n_jobs: 1
```

### Advanced AWS Support

When using AWS mode, the application with use `on-demand` EC2 instances from the `m5` series by default.

However, it is also possible to use `Spot` instances, specify a `max_hourly_price`, or customize your experience when using this mode in general.

All configuration points are grouped and documented under the `aws` yaml namespace in the main [config] file.

When setting  your own configuration, it is strongly recommended to first create your own `config.yaml` file as described in [Custom configuration](#custom-configuration).

_Example:_

A sample of a config file using Spot instances on a non-default region:
```yaml

aws:
  region: 'us-east-1'
  resource_files:
    - '{user}/config.yaml'
    - '{user}/frameworks.yaml'

  ec2:
    subnet_id: subnet-123456789   # subnet for account on us-east-1 region
    spot:
      enabled: true
      max_hourly_price: 0.40  # comment out to use default
```

## Add a benchmark

In this section, `benchmark` means a suite of datasets that can be used to feed any of the available frameworks, in combination with a set of constraints (time limit, cpus, memory) enforced by the application.

A benchmark definition will then consist in a [datasets definition](#datasets-definition) and a [constraints definition](#constraint-definition).

Each dataset must contain a training set and a test set. There can be multiple training/test splits, in which case each split is named a `fold`, so that the same dataset can be benchmarked multiple times using a different fold.

### Datasets definition

A dataset definition consists in a `yaml` file listing all the task/datasets that will be used for the complete benchmark, 
or as an OpenML suite.

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

Alternatively, you can run the benchmark on a single OpenML task without writing a benchmark definition:
```bash
python runbenchmark.py randomforest openml/t/59
```

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

#### OpenML suites

[OpenML] suites are a collection of OpenML tasks, for example <https://www.openml.org/s/218>.
You can run the benchmark on an openml suite directly, without defining the benchmark in a local file:
```bash
python runbenchmark.py randomforest openml/s/218
```

You can define a new OpenML suite yourself, for example through the Python API.
[This openml-python tutorial](https://openml.github.io/openml-python/master/examples/30_extended/suites_tutorial.html#sphx-glr-examples-30-extended-suites-tutorial-py)
explains how to build your own suite.
An advantage of using an OpenML suite is that sharing it is easy as the suite and its datasets can be accessed through APIs in many programming languages.

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
 1. to describe the framework and define which version will be used: `project`, `version`.
 1. to indicate the Python module with the integration code: `module` or `extends`.
 1. to pass optional parameters to the framework and/or the integration code: `params`.
 
Default framework definitions are defined in file `resources/frameworks.yaml` in lexicographic order, where `version` should be set to `stable`, which will point dynamically to the most recent official release available.

Frameworks that offer the possibility to test cutting edge version (e.g. nightly builds, `dev`/`master` repo, ...) can add an entry to `resources/frameworks_latest.yaml`, where `version` should be set to `latest`.

Maintainers of this repository try to regularly — ideally, every quarter — create a framework definition using frozen framework versions in order to favour the reproducibility of the published benchmarks.

Following the [custom configuration](#custom-configuration), it is possible to override and/or add a framework definitions by creating a `frameworks.yaml` file in your `user_dir`.

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
 1. add a brief description of the framework to the documentation in [docs/automl_overview](./automl_overview.md) following the same formatting as the other entries.
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


## Analyze the results

Except the logs, all the files generated by the application are in easy to process `csv` or `json` format, and they are all generated in a subfolder of the `output_dir` (by default, `./results`) unique for each benchmark run.

For example:
```text
results/randomforest.test.test.local.20201204T192714
|-- predictions
|   |-- cholesterol
|   |   |-- 0
|   |   |   |-- metadata.json
|   |   |   `-- predictions.csv
|   |   `-- 1
|   |       |-- metadata.json
|   |       `-- predictions.csv
|   |-- iris
|   |   |-- 0
|   |   |   |-- metadata.json
|   |   |   `-- predictions.csv
|   |   `-- 1
|   |       |-- metadata.json
|   |       `-- predictions.csv
|   `-- kc2
|       |-- 0
|       |   |-- metadata.json
|       |   `-- predictions.csv
|       `-- 1
|           |-- metadata.json
|           `-- predictions.csv
`-- scores
    |-- RandomForest.benchmark_test.csv
    `-- results.csv
```

### Results file format

Here is a sample `results.csv` file from a test run against the `RandomForest` framework:

```csv
id,task,framework,constraint,fold,result,metric,mode,version,params,tag,utc,duration,models,seed,info,acc,auc,balacc,logloss,mae,r2,rmse
openml.org/t/3913,kc2,RandomForest,test,0,0.865801,auc,local,0.23.2,{'n_estimators': 2000},,2020-12-04T19:27:46,3.2,2000,2633845682,,0.792453,0.865801,0.634199,0.350891,,,
openml.org/t/3913,kc2,RandomForest,test,1,0.86039,auc,local,0.23.2,{'n_estimators': 2000},,2020-12-04T19:27:52,3.0,2000,2633845683,,0.90566,0.86039,0.772727,0.406952,,,
openml.org/t/59,iris,RandomForest,test,0,0.126485,logloss,local,0.23.2,{'n_estimators': 2000},,2020-12-04T19:27:56,2.9,2000,2633845682,,0.933333,,0.933333,0.126485,,,
openml.org/t/59,iris,RandomForest,test,1,0.0271781,logloss,local,0.23.2,{'n_estimators': 2000},,2020-12-04T19:28:01,3.0,2000,2633845683,,1.0,,1.0,0.0271781,,,
openml.org/t/2295,cholesterol,RandomForest,test,0,44.3352,rmse,local,0.23.2,{'n_estimators': 2000},,2020-12-04T19:28:05,3.0,2000,2633845682,,,,,,35.6783,-0.014619,44.3352
openml.org/t/2295,cholesterol,RandomForest,test,1,55.3163,rmse,local,0.23.2,{'n_estimators': 2000},,2020-12-04T19:28:10,3.1,2000,2633845683,,,,,,43.1808,-0.0610752,55.3163
```
which gives in more readable format:
```text
                  id         task     framework constraint fold     result   metric   mode version                  params                  utc  duration models        seed       acc       auc    balacc   logloss      mae        r2     rmse
0  openml.org/t/3913          kc2  RandomForest       test    0   0.865801      auc  local  0.23.2  {'n_estimators': 2000}  2020-12-04T19:27:46       3.2   2000  2633845682  0.792453  0.865801  0.634199  0.350891      NaN       NaN      NaN
1  openml.org/t/3913          kc2  RandomForest       test    1   0.860390      auc  local  0.23.2  {'n_estimators': 2000}  2020-12-04T19:27:52       3.0   2000  2633845683  0.905660  0.860390  0.772727  0.406952      NaN       NaN      NaN
2    openml.org/t/59         iris  RandomForest       test    0   0.126485  logloss  local  0.23.2  {'n_estimators': 2000}  2020-12-04T19:27:56       2.9   2000  2633845682  0.933333       NaN  0.933333  0.126485      NaN       NaN      NaN
3    openml.org/t/59         iris  RandomForest       test    1   0.027178  logloss  local  0.23.2  {'n_estimators': 2000}  2020-12-04T19:28:01       3.0   2000  2633845683  1.000000       NaN  1.000000  0.027178      NaN       NaN      NaN
4  openml.org/t/2295  cholesterol  RandomForest       test    0  44.335200     rmse  local  0.23.2  {'n_estimators': 2000}  2020-12-04T19:28:05       3.0   2000  2633845682       NaN       NaN       NaN       NaN  35.6783 -0.014619  44.3352
5  openml.org/t/2295  cholesterol  RandomForest       test    1  55.316300     rmse  local  0.23.2  {'n_estimators': 2000}  2020-12-04T19:28:10       3.1   2000  2633845683       NaN       NaN       NaN       NaN  43.1808 -0.061075  55.3163

```

Here is a short description of each column:
- `id`: a identifier for the dataset used in this result. For convenience, we use the link to the OpenML task by default.
- `task`: the task name as defined in the benchmark definition.
- `framework`: the framework name as defined in the framework definition.
- `fold`: the dataset fold being used for this job. Usually, we're using 10 folds, so the fold varies from 0 to 9.
- `result`: the result score, this is the score for the metric that the framework was trying to optimize. For example, for binary classification, the default metrics defined in `resources/config.yaml` are `binary: ['auc', 'acc']`; this means that the frameworks should try to optimize `auc` and the final `auc` score will become the `result` value, the other metrics (here `acc`) are then computed for information.
- `mode`: one of `local`, `docker`, `aws`, `aws+docker`: tells where/how the job was executed.
- `version`: the version of the framework being benchmarked.
- `params`: if any, a JSON representation of the params defined in the framework definition. This allows to see clearly if some tuning was done for example.
- `tag`: the branch tag of the `automlbenchmark` app that was running the job.
- `utc`: the UTC timestamp at the job completion.
- `duration`: the training duration: the framework integration is supposed to provide this information to ensure that it takes only into account the time taken by the framework itself. When benchmarking large data, the application can use a significant amount of time to prepare the data: this additional time doesn't appear in this `duration` column.
- `models`: for some frameworks, it is possible to know how many models in total were trained by the AutoML framework. 
- `seed`: the seed or random state passed to the framework. With some frameworks, it is enough to obtain reproducible results. Note that the seed can be specified at the command line using `-Xseed=` arg (for example `python randomforest -Xseed=1452956522`): when there are multiple folds, the seed is then incremented by the fold number.
- `info`: additional info in text format, this usually contains error messages if the job failed.
- `acc`, `auc`, `logloss` metrics: all the metrics that were computed based on the generated predictions. For each job/row, one of them matches the `result` column, the others are purely informative. Those additional metric columns are simply added in alphabetical order.

### Predictions

For each training, the framework integration must generate a predictions file that will be used by the application to compute the scores.

This predictions file:
- must be saved under the `predictions` subfolder as shown [above](#analyze-the-results).
- follow the naming convention: `{framework}_{task}_{fold}.csv`.
- must be in `CSV` format with first row as header.
- for classification problems, the header should look like: `*class_labels | predictions | truth`, with
  - `*class_labels` meaning an alphabetically ordered list of class/target labels, each column containing the probabilities for the corresponding class. If the framework doesn't provide those probabilities, then the framework integration should provide pseudo-probabilities, using `1` for the predicted value and `0` for other classes.
  - `predictions` column containing the predictions of the test predictor data (`test.X`) by the model trained by the framework,
  - `truth` being the test target data (`test.y`).
- for regression problems, the header should look like `predictions | truth`, with
  - `predictions` column containing the predictions of the test predictor data (`test.X`) by the model trained by the framework, 
  - `truth` being the test target data (`test.y`). 
  
_Examples_:

Predictions sample on `binary` classification (`kc2`):

in CSV format:
```csv
no,yes,predictions,truth
0.965857617846013,0.034142382153998944,no,no
0.965857617846013,0.034142382153998944,no,no
0.5845,0.4155,no,no
0.6795,0.3205,no,no
0.965857617846013,0.034142382153998944,no,no
```
as table:

| no                | yes                  | predictions | truth | 
|-------------------|----------------------|-------------|-------| 
| 0.965857617846013 | 0.034142382153998944 | no          | no    | 
| 0.965857617846013 | 0.034142382153998944 | no          | no    | 
| 0.5845            | 0.4155               | no          | no    | 
| 0.6795            | 0.3205               | no          | no    | 
| 0.965857617846013 | 0.034142382153998944 | no          | no    | 


Predictions sample on `multiclass` classification (`iris`):

in CSV format:
```csv
Iris-setosa,Iris-versicolor,Iris-virginica,predictions,truth
1.0,0.0,0.0,Iris-setosa,Iris-setosa
0.9715,0.028,0.0005,Iris-setosa,Iris-setosa
1.0,0.0,0.0,Iris-setosa,Iris-setosa
1.0,0.0,0.0,Iris-setosa,Iris-setosa
1.0,0.0,0.0,Iris-setosa,Iris-setosa
0.0,1.0,0.0,Iris-versicolor,Iris-versicolor
0.0,0.976,0.024,Iris-versicolor,Iris-versicolor
0.0,0.994,0.006,Iris-versicolor,Iris-versicolor
0.0,0.9925,0.0075,Iris-versicolor,Iris-versicolor
0.0,0.995,0.005,Iris-versicolor,Iris-versicolor
0.0,0.829,0.171,Iris-versicolor,Iris-virginica
0.0,0.008,0.992,Iris-virginica,Iris-virginica
0.0,0.0005,0.9995,Iris-virginica,Iris-virginica
0.0,0.0015,0.9985,Iris-virginica,Iris-virginica
0.0,0.0395,0.9605,Iris-virginica,Iris-virginica
```
as table:

| Iris-setosa | Iris-versicolor | Iris-virginica | predictions     | truth           | 
|-------------|-----------------|----------------|-----------------|-----------------| 
| 1.0         | 0.0             | 0.0            | Iris-setosa     | Iris-setosa     | 
| 0.9715      | 0.028           | 0.0005         | Iris-setosa     | Iris-setosa     | 
| 1.0         | 0.0             | 0.0            | Iris-setosa     | Iris-setosa     | 
| 1.0         | 0.0             | 0.0            | Iris-setosa     | Iris-setosa     | 
| 1.0         | 0.0             | 0.0            | Iris-setosa     | Iris-setosa     | 
| 0.0         | 1.0             | 0.0            | Iris-versicolor | Iris-versicolor | 
| 0.0         | 0.976           | 0.024          | Iris-versicolor | Iris-versicolor | 
| 0.0         | 0.994           | 0.006          | Iris-versicolor | Iris-versicolor | 
| 0.0         | 0.9925          | 0.0075         | Iris-versicolor | Iris-versicolor | 
| 0.0         | 0.995           | 0.005          | Iris-versicolor | Iris-versicolor | 
| 0.0         | 0.829           | 0.171          | Iris-versicolor | Iris-virginica  | 
| 0.0         | 0.008           | 0.992          | Iris-virginica  | Iris-virginica  | 
| 0.0         | 0.0005          | 0.9995         | Iris-virginica  | Iris-virginica  | 
| 0.0         | 0.0015          | 0.9985         | Iris-virginica  | Iris-virginica  | 
| 0.0         | 0.0395          | 0.9605         | Iris-virginica  | Iris-virginica  | 

Predictions sample on `regression` (`cholesterol`):

in CSV format:
```csv
predictions,truth
241.204,207.0
248.9575,249.0
302.278,268.0
225.9215,234.0
226.6995,201.0
```
as table:

| predictions | truth | 
|-------------|-------| 
| 241.204     | 207.0 | 
| 248.9575    | 249.0 | 
| 302.278     | 268.0 | 
| 225.9215    | 234.0 | 
| 226.6995    | 201.0 | 


### Extract more information

For some frameworks, it is also possible to extract more detailed information, in the form of `artifacts` that are saved after the training.

Examples of those artifacts are:
- logs generated by the framework.
- models or descriptions of the models trained by the framework.
- predictions for each of the model trained by the AutoML framework.
- ...

By default, those artifacts are not saved, and all frameworks don't provide the same artifacts, that's why the list of the artifacts that should be extracted can only be specified in the framework definition using the conventional `_save_artifacts` param:

_Examples:_

```yaml
autosklearn_debug:
  extends: autosklearn
  params:
    _save_artifacts: ['models']  # will save models descriptions under the `models` subfolder

H2OAutoML_debug:
  extends: H2OAutoML
  params:
    _save_artifacts: ['leaderboard', 'logs', 'models']  # will save the leaderboard and models under the `models` subfolder, and the H2O logs under `logs` subfolder.

TPOT_debug:
  extends: TPOT
  params:
    _save_artifacts: ['models']
```

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

### Framework setup fails
If the setup fails, first note that only the following OS are fully supported:
- Ubuntu 18.04

The setup is created for Debian-based linux environments, and macOS (most frameworks can be installed on macOS, ideally with `brew` installed, but there may be a few exceptions), so it may work with other Linux environments not listed above (e.g. Debian, Ubuntu 20.04, ...).
The best way to run benchmarks on non-supported OS, is to use the docker mode.

If the setup fails on a supported environment, please try the following:
- force the setup: see above.
- ensure that the same framework is not set up multiple times in parallel on the same machine:
  - first use `python runbenchmark.py MyFramework -s only` on one terminal.
  - then you can trigger multiple `python runbenchmark.py MyFramework ...` (without `-s` option) in parallel.
- delete the `lib` and `venv` folders, if present, under the given framework folder (e.g. `frameworks/MyFramework`), and try the setup again.


[README]: ./README.md
[OpenML]: https://openml.org
[ARFF]: https://waikato.github.io/weka-wiki/formats_and_processing/arff_stable/
[CSV]: https://tools.ietf.org/html/rfc4180
[Docker]: https://docs.docker.com/
[config]: ../resources/config.yaml
