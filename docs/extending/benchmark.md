# Benchmark

Benchmarks are collections of machine learning tasks, where each task is a dataset
with associated information on train/test splits used to evaluate the model.
These tasks can be defined in a `yaml` file or on [OpenML](https://www.openml.org).
Both options allow for defining a benchmark of one or more datasets.
It is even possible to reference to OpenML tasks from a benchmark file.

!!! note "Supported Datasets"
    
    Currently, the AutoML benchmark only supports definitions of tabular datasets for
    classification, regression, and time series forecasting. The time series forecasting
    support is in an early stage, subject to change, and not supported through OpenML.

## Defining a Benchmark on OpenML
Especially when performing a benchmark evaluation to be used in a publication, we
recommend the use of OpenML for the definition of the benchmark if possible. This
ensures that other users can run your benchmark out of the box, without any required
additional files. OpenML also provides a lot of meta-data about the datasets which is
also accessible through [APIs](https://www.openml.org/apis) in various programming 
languages. We recommend using the [`openml-python`](https://openml.github.io/openml-python)
Python library as it is the most comprehensive of the OpenML libraries.

Defining a benchmark on OpenML requires the following steps:

 - [Upload a dataset](https://openml.github.io/openml-python/main/examples/30_extended/create_upload_tutorial.html#sphx-glr-examples-30-extended-create-upload-tutorial-py). 
   A dataset is the tabular data, alongside meta-data like its name,
   authors, and license. OpenML will also automatically extract meta-data about the
   datasets, such as feature types, class balance, or dataset size. After uploading the
   dataset, it will receive an identifier (`ID`) and should be visible on the OpenML
   website on `www.openml.org/d/ID`.
 - [Define a task](https://openml.github.io/openml-python/main/generated/openml.tasks.create_task.html#openml.tasks.create_task). 
   A task defines how to evaluate a model on a given dataset, for example
   "10-fold cross-validation optimizing AUC". OpenML will generate splits for the 10-fold
   cross-validation procedure which means that anyone using this task definition can 
   perform the experiment with the exact same splits easily.
 - [Define a benchmark suite](https://openml.github.io/openml-python/main/examples/30_extended/suites_tutorial.html#sphx-glr-examples-30-extended-suites-tutorial-py). 
   On a technical level, a benchmarking suite is nothing more than a collection of tasks. 
   You can add a description that details the purpose of the benchmarking suite, or any 
   information that users should be aware of before using the suite.

When a task or benchmark suite is available on OpenML, it can be directly referred to
for the `benchmark` parameter of `runbenchmark.py` as `openml/s/ID` for suites and 
`openml/t/ID` for tasks, where `ID` is to be replaced with the OpenML identifier of the
object. For example, `openml/t/59` refers to [task 59](https://www.openml.org/t/59), 
which is 10-fold cross-validation on the [iris dataset](https://www.openml.org/d/61).

## Defining a Benchmark with a File

When defining a benchmark with a `yaml` file, the `yaml` will contain information about
tasks that are located either on disk or on OpenML. We make a few default benchmarks
available in our [`resources/benchmarks`](GITHUB/resources/benchmarks) folder:

 * `test`: a collection of three small datasets covering regression, binary classification, 
    and multiclass classification. This makes it incredibly useful for small tests and
    fast feedback on whether the software runs without error.
 * `validation`: a collection of datasets which have different edge cases, such as a
    very wide dataset, datasets with missing or non-numerical values, and more. This
    typically produces most errors you might also encounter when running larger 
    benchmarks.
 * `timeseries`: a benchmark for testing time series forecasting integration (experimental).

Below is an excerpt from the `test.yaml` file:

```yaml
- name: kc2
  openml_task_id: 3913
  description: "binary test dataset"
```

When writing your own benchmark definition, it needs to be discoverable by the benchmark.
A good place to do this would be adding a `benchmarks` directory to your benchmark
configuration directory (`~/.config/automlbenchmark` by default) and updating your
[custom configuration](../../using/configuration/#custom-configurations) by adding:

```yaml
benchmarks:                     
  definition_dir:               
    - '{root}/resources/benchmarks'
    - '{user}/resources/benchmarks'
```

Each task must have a name that is unique in the definition file (case-insensitive),
this name will also be used as identifier (e.g., in the results files).
Additionally, the file must have a description of where to find the dataset files
and splits. When you have a task already on OpenML, you can directly reference it with
`openml_task_id` to define the dataset and splits. Alternatively, you can use local files.

It is also possible to benchmark your own datasets that you can not or do not want to
upload to OpenML. The data files should be in `arff` or `csv` format and contain at least 
one file for training data and one file for test data. When working with multiple files,
it is useful to use an archive (`.zip`, `.tar`, `.tgz`, `.tbz`) or directory structure. 
Use the following naming convention to allow the AutoML benchmark to infer what each file represents:

    - if there's only one file for training and one for test, they should be named `{name}_train.csv` and `{name}_test.csv` (in case of CSV files).
    - if there are multiple `folds`, they should follow a similar convention: `{name}_train_0.csv`, `{name}_test_0.csv``, {name}_train_1.csv`, `{name}_test_1.csv`, ...

Examples:

=== "Single Fold CSV"

    ```yaml
    - name: example_csv
      dataset:
        train: /path/to/data/ExampleTraining.csv
        test:  /path/to/data/ExampleTest.csv
        target: TargetColumn
      folds: 1
    ```

=== "Multiple Folds CSV"

    ```yaml
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
    ```

=== "Directory"

    It is important that the files in the directory follow the naming convention described above.

    ```yaml
    - name: example_dir
      dataset: 
        path: /path/to/data
        target: TargetColumn
      folds: 1
    ```

=== "Archive"

    It is important that the files in the archive follow the naming convention described above.

    ```yaml
    - name: example_archive
      dataset:
        path: /path/to/archive.zip
        target: TargetColumn
      folds: 3
    ```

=== "Remote Files"

    The remote file may also be an archive. If that is the case, it is important that 
    the files in the archive follow the naming convention described above.

    ```yaml
    - name:  example_csv_http
      dataset:
        train: https://my.domain.org/data/ExampleTraining.csv
        test:  https://my.domain.org/data/ExampleTest.csv
        target: TargetColumn
      folds: 1
    ```

    Remote files are downloaded to the `input_dir` folder and archives are decompressed 
    there as well. You can change the value of this folder in your 
    [custom config.yaml file](../../using/configuration/#custom-configurations) 
    or specify it at the command line with the `-i` or `--indir` argument 
    (by default, it points to the `~/.openml/cache` folder).


The `target` attribute is optional but recommended. If not set, it will resolve to the 
column `target` or `class` if present, and the last column otherwise.

You can even make use of the [special directives](../../using/configuration/#custom-configurations) like `{user}`.

```yaml
- name: example_relative_to_user_dir
  dataset:
    train: "{user}/data/train.csv"
    test: "{user}/data/test.csv"
```

After creating a benchmark definition, e.g. `~/.config/automlbenchmark/benchmarks/my_benchmark.yaml`,
it can then be referenced when running `runbenchmark.py`: `python runbenchmark.py FRAMEWORK my_benchmark`.

## Defining a Time Series Forecasting Dataset

!!! warning "Time Series Forecasting should be considered experimental"

    Time series forecasting support should be considered experimental and is currently
    only supported with the AutoGluon integration.

Benchmark definitions for time series datasets work in much the same way, but there are
some additional fields and requirements to a valid time series dataset.

First, the dataset must be stored as a single csv file in 
[long format](https://doc.dataiku.com/dss/latest/time-series/data-formatting.html#long-format) 
and must include 3 columns:

  - `id_column`: An indicator column that specifies to which time series the sample belongs by a unique id.
    The default expected name of this column is "item_id".
  - `timestamp_column`: A column with the timestamp of the observation.
    The default expected name of this column is "timestamp".
  - `target`: A column with the target value of the time series

Additionally, the data must satisfy the following criteria:

 - The shortest time series in the dataset must have length of at least `folds * forecast_horizon_in_step + 1` (see [Generated Folds](#generated-folds)).
 - Time series may have different lengths or have different starting timestamps, 
   but must have the same frequency.
 - All time series must have regular timestamp index, i.e., it must have an observation
   for each time step from start to end.

If the `id_column` or `timestamp_column` names are not the default expected ones,
they must be explicitly stated in the definition, as can be seen in the examples below.
Moreover, the definition must also contain the following fields:

  - `path`: a local or remote path to the CSV file with time series data.
  - `freq`: a [pandas-compatible frequency string](https://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases) 
    that denotes the frequency of the time series. For example, `D` for daily, `H` for hourly, or `15min` for 15-minute frequency.
  - `forecast_horizon_in_steps`: a positive integer denoting how many future time series values need to be predicted.
  - `seasonality`: a positive integer denoting the seasonal period of the data, measured in steps. 
    This parameter is used for computing metrics like [mean absolute scaled error](https://en.wikipedia.org/wiki/Mean_absolute_scaled_error#Seasonal_time_series) (denoted as *m* on Wikipedia).


=== "Default Column Names"

    Given a file at `path.to/data.csv` that contains two time series with daily frequency, 
    `A` with three observations and `B` with four observations:
    
    | item_id |	timestamp |	target |
    |---------|-----------|--------:|
    | A       |	2020-01-01|	2.0    |
    | A       |	2020-01-02|	1.0    |
    | A       |	2020-01-03|	5.0    |
    | B       |	2019-05-02|	8.0    |
    | B       |	2019-05-03|	2.0    |
    | B       |	2019-05-04|	1.0    |
    | B       |	2019-05-05|	9.0    |

    When we specify the fields outlined above, then the respective task definition may 
    look like the one below. Note that we do not have to specify `id_column` or 
    `timestamp_column` as their names match the default expected value.
    
    ```yaml
    - name: example_time_series_data
      dataset:
        path: /path/to/data.csv
        freq: D
        forecast_horizon_in_steps: 1
        seasonality: 7
        target: target
      folds: 1
    ```

    

=== "Non-default Column Names"

    Given a file at `path.to/data.csv` that contains two time series with daily frequency, 
    `A` with three observations and `B` with four observations. It is identical to
    the example "default column values", but the header provides different column names:
    
    | Product |	Date |	Value |
    |---------|-----------|--------:|
    | A       |	2020-01-01|	2.0    |
    | A       |	2020-01-02|	1.0    |
    | A       |	2020-01-03|	5.0    |
    | B       |	2019-05-02|	8.0    |
    | B       |	2019-05-03|	2.0    |
    | B       |	2019-05-04|	1.0    |
    | B       |	2019-05-05|	9.0    |

    When we specify the fields outlined above, then the respective task definition may 
    look like the one below. Note that we do *have to* specify `id_column` or 
    `timestamp_column` as their names do not match the default expected value. If left 
    unspecified, the benchmark tool will raise an error.
    
    ```yaml
    - name: example_time_series_data
      dataset:
        path: /path/to/data.csv
        freq: D
        forecast_horizon_in_steps: 1
        seasonality: 7
        id_column: Product
        timestamp_column: Date
        target: Value
      folds: 1
    ```

    

### Generated Folds

AMLB automatically generates the train and test splits from the raw data depending 
on the chosen `forecast_horizon_in_steps` and `folds` parameters. Assuming 
`forecast_horizon_in_steps = K` and `folds = n`, and each time series has length `n * K`,
the folds will be generated as follows:

  rows | fold 0 | fold 1 | ... | fold (n-2) | fold (n-1)
  -- | -- | -- | -- | -- | --
  1..K | train | train | ... | train | train
  K..2K | train | train | ... | train | test
  2..3K | train  | train | ... | test |
  ... |   |   |     |  
  (n-2)K...(n-1)K | train  |  test   | |
  (n-1)K...nK | test  |    | |

As a consequence, the shortest time series in the dataset must have length of at least 
`folds * forecast_horizon_in_step + 1`.

!!! warning "This is still batch learning!"
    
    It is important to note that the model does not carry over between folds, each fold
    the model is trained from scratch on the available training data. As such, it is
    still batch learning, as opposed to [train-then-test](https://scikit-multiflow.readthedocs.io/en/stable/user-guide/core-concepts.html) 
    (or prequential) evaluation where a single model is continuously updated instead.
    