# HOW-TO

## Run a benchmark
see [README] `Quickstart` section for basic commands.

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

Following the [custom configuration](#custom-configuration), it is possible to replace and/or add constraints by creating the following `constraints.yaml` file in your `user_dir`:

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

The new constraints can then be used on the command line when executing the benchmark:
```bash
python runbenchmark.py randomforest validation 1h16c
```

#### Add a default benchmark

#### Add a custom benchmark

## Add an AutoML framework

### Framework definition

### Framework integration

#### Recommended structure

#### Frameworks with Python API

##### Frameworks requiring a dedicated virtual env

#### Other Frameworks



#### Add a default framework

#### Add a custom framework

## Troubleshooting guide

### Logs

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

[README]: ./README.md
[OpenML]: https://openml.org
[ARFF]: https://waikato.github.io/weka-wiki/formats_and_processing/arff_stable/
[CSV]: https://tools.ietf.org/html/rfc4180
