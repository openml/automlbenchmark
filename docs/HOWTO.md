---
---
# HOW-TO

## Run a benchmark
see [README], `Quickstart` section for basic commands.

### Custom configuration

Default configurations are all defined with some explanations in the `resources/config.yaml` file.

If you want to change those configurations, it is not recommended to edit the file above, but rather to create your own `config.yaml` file under the `user_dir` (by default `~/.config/automlbenchmark`). The application will automatically load this file and use to override the defaults.

_Example of config.yaml:_
```yaml
---
input_dir: ~/data

frameworks:
  definition_file:  # this allows to use default frameworks + those defined in your custom frameworks.yaml.
    - '{root}/resources/frameworks.yaml'
    - '{user}/frameworks.yaml'

benchmarks:
  definition_dir:  # this allows to put your custom benchmark definitions under {user}/benchmarks.
    - '{user}/benchmarks'
    - '{root}/resources/benchmarks'

aws:
  resource_files:  # this allows to automatically upload custom config + frameworks to the running instance (benchmark files are always uploaded).
    - '{user}/config.yaml'
    - '{user}/frameworks.yaml'

  use_docker: true  # you can decide to always use the prebuilt docker images on AWS.
```  

#### Run a framework with different (hyper-)parameters

Framework definitions accept a `params` dictionary for pass-through parameters that are directly accessible from the framework `exec.py` file that does the AutoML training.

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

**NOTE:** by convention, the param names starting with `_` are filtered out (they are not passed to the classifier/regressor) but are used for custom logic in the `exec.py`.

_Example:_
 
In the definition below, the `_n_jobs` param is handled by custom code in `RandomForest/exec.py` (here it overrides the default `n_jobs` automatically calculated by the application: using all assigned cores).
```yaml
RandomForest:
  version: '0.21.3'
  project: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
  params:
    n_estimators: 2000
    _n_jobs: 1
```

## Add a benchmark

By benchmark here, we mean a suite of datasets that can be used to feed any of the available frameworks, in combination with a set of constraints (time limit, cpus, memory) enforced by the application.

Each dataset must contain a training set and a test set. There can be various splits between the training set and the test one, in which case each split is named a `fold`, so that the same dataset can be benchmarked multiple times using a different fold. 

### Benchmark definition

A benchmark definition consists in a `yaml` file listing all the task/datasets that will be used when running the complete benchmark.

Each task/dataset must have a `name` that must be unique in the given definition file, and that will be used as an identifier, for example in the results.

This `name` can also be used on the command line (`-t` or `--task` argument) when we don't want to execute the full benchmark, but just a subset, often in combination with a specific fold (`-f` or `--fold` argument):
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
where `openml_task_id` allows accessing the OpenML task at `https://www.openml.org/t/{openml_task_id}` (in this example: https://www.openml.org/t/9910). 


#### OpenML studies

[OpenML] studies are a collection of OpenML tasks, for example https://www.openml.org/s/218.
The application doesn't directly support OpenML studies for now: they need to be converted into a proper benchmark definition file including all the tasks from the study, but we're thinking about improving this: see https://github.com/openml/automlbenchmark/issues/61.

#### File datasets

It is also possible to benchmark the supported frameworks using your own datasets, as soon as they follow some requirements:
- Each dataset must contain at least one file for training data and one file for test data.
- The data files should be in one of the currently supported format: [ARFF], [CSV] (ideally with header).
- If the dataset is represented as an archive (.zip, .tar, .tgz, .tbz) or a directory, then the data files must follow this naming convention to be detected correctly:
  - if there's only one file for training and one for test, they should be named `{name}_train.csv` and `{name}_test.csv` (in case of CSV files).
  - if there are multiple `folds`, they should follow a similar convention: `{name}_train_0.csv`, `{name}_test_0.csv``, {name}_train_1.csv`, `{name}_test_1.csv`, ...

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
  0. looking for a column named `target` of `class`.
  0. using the last column as a fallback.
- the `folds` attribute is also optional but recommended for those datasets as the default value is `folds=10` (common default with openml datasets), so if you don't have that many folds for your custom datasets, it is better to declare it explicitly here.
- Remote files are downloaded to the `input_dir` folder and archives are decompressed there as well, so you may want to change the value of this folder in your [custom config.yaml file](#custom-configuration) or specify it at the command line with the `-i` or `--indir` argument (by default, it points to the `~/.openml/cache` folder).

### Constraints definition

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
