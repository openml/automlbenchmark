---
title: Getting Started
description: A short tutorial on installing the software and running a simple benchmark.
---

# Getting Started

The AutoML Benchmark is a tool for benchmarking AutoML frameworks on tabular data.
It automates the installation of AutoML frameworks, passing it data, and evaluating
their predictions. 
[Our paper](https://arxiv.org/pdf/2207.12560.pdf) describes the design and showcases 
results from an evaluation using the benchmark. 
This guide goes over the minimum steps needed to evaluate an
AutoML framework on a toy dataset.

## Installation
These instructions assume that [Python 3.9 (or higher)](https://www.python.org/downloads/) 
and [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) are installed,
and are available under the alias `python` and `git`, respectively. We recommend
[Pyenv](https://github.com/pyenv/pyenv) for managing multiple Python installations,
if applicable. We support Ubuntu 22.04, but many linux and MacOS versions likely work
(for MacOS, it may be necessary to have [`brew`](https://brew.sh) installed).

First, clone the repository:

```bash
git clone https://github.com/openml/automlbenchmark.git --branch stable --depth 1
cd automlbenchmark
```

Create a virtual environments to install the dependencies in:

=== ":simple-linux: Linux"

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

=== ":material-apple: MacOS"

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

=== ":simple-windows: Windows"

    ```bash
    python -m venv ./venv
    venv/Scripts/activate
    ```

Then install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```


??? windows "Note for Windows users"

    The automated installation of AutoML frameworks is done using shell script,
    which doesn't work on Windows. We recommend you use
    [Docker](https://docs.docker.com/desktop/install/windows-install/) to run the
    examples below. First, install and run `docker`. 
    Then, whenever there is a `python runbenchmark.py ...` 
    command in the tutorial, add `-m docker` to it (`python runbenchmark.py ... -m docker`).

??? question "Problem with the installation?"

    On some platforms, we need to ensure that requirements are installed sequentially.
    Use `xargs -L 1 python -m pip install < requirements.txt` to do so. If problems 
    persist, [open an issue](https://github.com/openml/automlbenchmark/issues/new) with
    the error and information about your environment (OS, Python version, pip version).


## Running the Benchmark

To run a benchmark call the `runbenchmark.py` script specifying the framework to evaluate.
See [integrated frameworks](WEBSITE/frameworks.html) for a list of supported frameworks, or the [adding a frameworking](extending/framework.md) page on how to add your own.

### Example: a test run with Random Forest
Let's try evaluating the `RandomForest` baseline, which uses [scikit-learn](https://scikit-learn.org/stable/)'s random forest:

=== ":simple-linux: Linux"

    ```bash
    python runbenchmark.py randomforest 
    ```

=== ":material-apple: MacOS"

    ```bash
    python runbenchmark.py randomforest 
    ```

=== ":simple-windows: Windows"
    As noted above, we need to install the AutoML frameworks (and baselines) in
    a container. Add `-m docker` to the command as shown:
    ```bash
    python runbenchmark.py randomforest -m docker
    ```

    !!! warning "Important"
        Future example usages will only show invocations without `-m docker` mode,
        but Windows users will need to run in some non-local mode.

After running the command, there will be a lot of output to the screen that reports
on what is currently happening. After a few minutes final results are shown and should 
look similar to this:

```
Summing up scores for current run:
               id        task  fold    framework constraint     result      metric  duration      seed
openml.org/t/3913         kc2     0 RandomForest       test   0.865801         auc      11.1 851722466
openml.org/t/3913         kc2     1 RandomForest       test   0.857143         auc       9.1 851722467
  openml.org/t/59        iris     0 RandomForest       test  -0.120755 neg_logloss       8.7 851722466
  openml.org/t/59        iris     1 RandomForest       test  -0.027781 neg_logloss       8.5 851722467
openml.org/t/2295 cholesterol     0 RandomForest       test -44.220800    neg_rmse       8.7 851722466
openml.org/t/2295 cholesterol     1 RandomForest       test -55.216500    neg_rmse       8.7 851722467
```

The result denotes the performance of the framework on the test data as measured by
the metric listed in the metric column. The result column always denotes performance 
in a way where higher is better (metrics which normally observe "lower is better" are
converted, which can be observed from the `neg_` prefix).

While running the command, the AutoML benchmark performed the following steps:

 1. Create a new virtual environment for the Random Forest experiment. 
    This environment can be found in `frameworks/randomforest/venv` and will be re-used 
    when you perform other experiments with `RandomForest`.
 2. It downloaded datasets from [OpenML](https://www.openml.org) complete with a 
    "task definition" which specifies [cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html) folds.
 3. It evaluated `RandomForest` on each (task, fold)-combination in a separate subprocess, where:
    1. The framework (`RandomForest`) is initialized.
    2. The training data is passed to the framework for training.
    3. The test data is passed to the framework to make predictions on.
    4. It passes the predictions back to the main process
 4. The predictions are evaluated and reported on. They are printed to the console and 
    are stored in the `results` directory. There you will find:
    1. `results/results.csv`: a file with all results from all benchmarks conducted on your machine.
    2. `results/randomforest.test.test.local.TIMESTAMP`: a directory with more information about the run,
        such as logs, predictions, and possibly other artifacts.

!!! info "Docker Mode" 

    When using docker mode (with `-m docker`) a docker image will be made that contains
    the virtual environment. Otherwise, it functions much the same way.

### Important Parameters

As you can see from the results above, the  default behavior is to execute a short test
benchmark. However, we can specify a different benchmark, provide different constraints,
and even run the experiment in a container or on AWS. There are many parameters
for the `runbenchmark.py` script, but the most important ones are:

`Framework (required)`

: The AutoML framework or baseline to evaluate and is not case-sensitive. See
  [integrated frameworks](WEBSITE/frameworks.html) for a list of supported frameworks. 
  In the above example, this benchmarked framework `randomforest`.

`Benchmark (optional, default='test')`

: The benchmark suite is the dataset or set of datasets to evaluate the framework on.
  These can be defined as on [OpenML](https://www.openml.org) as a [study or task](extending/benchmark.md#defining-a-benchmark-on-openml) 
  (formatted as `openml/s/X` or `openml/t/Y` respectively) or in a [local file](extending/benchmark.md#defining-a-benchmark-with-a-file).
  The default is a short evaluation on two folds of `iris`, `kc2`, and `cholesterol`.

`Constraints (optional, default='test')`

: The constraints applied to the benchmark as defined by default in [constraints.yaml](GITHUB/resources/constraints.yaml).
  These include time constraints, memory constrains, the number of available cpu cores, and more.
  Default constraint is `test` (2 folds for 10 min each). 

    !!! warning "Constraints are not enforced!"
        These constraints are forwarded to the AutoML framework if possible but, except for
        runtime constraints, are generally not enforced. It is advised when benchmarking
        to use an environment that mimics the given constraints.

    ??? info "Constraints can be overriden by `benchmark`"
        A benchmark definition can override constraints on a task level.
        This is useful if you want to define a benchmark which has different constraints
        for different tasks. The default "test" benchmark does this to limit runtime to
        60 seconds instead of 600 seconds, which is useful to get quick results for its
        small datasets. For more information, see [defining a benchmark](#ADD-link-to-adding-benchmark).

`Mode (optional, default='local')`

:  The benchmark can be run in four modes:

     * `local`: install a local virtual environment and run the benchmark on your machine.
     * `docker`: create a docker image with the virtual environment and run the benchmark in a container on your machine. 
                 If a local or remote image already exists, that will be used instead. Requires [Docker](https://docs.docker.com/desktop/).
     * `singularity`: create a singularity image with the virtual environment and run the benchmark in a container on your machine. Requires [Singularity](https://docs.sylabs.io/guides/3.5/user-guide/introduction.html).
     * `aws`: run the benchmark on [AWS EC2](https://aws.amazon.com/free/?trk=b3f93e34-c1e0-4aa9-95f8-6d2c36891d8a&sc_channel=ps&ef_id=CjwKCAjw-7OlBhB8EiwAnoOEk0li05IUgU9Ok2uCdejP22Yr7ZuqtMeJZAdxgL5KZFaeOVskCAsknhoCSjUQAvD_BwE:G:s&s_kwcid=AL!4422!3!649687387631!e!!g!!aws%20ec2!19738730094!148084749082&all-free-tier.sort-by=item.additionalFields.SortRank&all-free-tier.sort-order=asc&awsf.Free%20Tier%20Types=*all&awsf.Free%20Tier%20Categories=*all) instances.
              It is possible to run directly on the instance or have the EC2 instance run in `docker` mode.
              Requires valid AWS credentials to be configured, for more information see [Running on AWS](#ADD-link-to-aws-guide).


For a full list of parameters available, run:

```
python runbenchmark.py --help
```

### Example: AutoML on a specific task and fold

The defaults are very useful for performing a quick test, as the datasets are small
and cover different task types (binary classification, multiclass classification, and 
regression). We also have a ["validation" benchmark](GITHUB/resources/benchmarks/validation.yaml)
suite for more elaborate testing that also includes missing data, categorical data, 
wide data, and more. The benchmark defines 9 tasks, and evaluating two folds with a
10-minute time constraint would take roughly 3 hours (=9 tasks * 2 folds * 10 minutes,
plus overhead). Let's instead use the `--task` and `--fold` parameters to run only a
specific task and fold in the `benchmark` when evaluating the 
[flaml](https://microsoft.github.io/FLAML/) AutoML framework:

```
python runbenchmark.py flaml validation test -t eucalyptus -f 0
```

This should take about 10 minutes plus the time it takes to install `flaml`.
Results should look roughly like this:

```
Processing results for flaml.validation.test.local.20230711T122823
Summing up scores for current run:
               id       task  fold framework constraint    result      metric  duration       seed
openml.org/t/2079 eucalyptus     0     flaml       test -0.702976 neg_logloss     611.0 1385946458
```

Similarly to the test run, you will find additional files in the `results` directory.


### Example: Benchmarks on OpenML

In the previous examples, we used benchmarks which were defined in a local file
([test.yaml](GITHUB/resources/benchmarks/test.yaml) and 
[validation.yaml](GITHUB/resources/benchmarks/validation.yaml), respectively). 
However, we can also use tasks and
benchmarking suites defined on OpenML directly from the command line. When referencing
an OpenML task or suite, we can use `openml/t/ID` or `openml/s/ID` respectively as 
argument for the benchmark parameter. Running on the [iris task](https://openml.org/t/59):

```
python runbenchmark.py randomforest openml/t/59
```

or on the entire [AutoML benchmark classification suite](https://openml.org/s/271) (this will take hours!):

```
python runbenchmark.py randomforest openml/s/271
```

!!! info "Large-scale Benchmarking"

    For large scale benchmarking it is advised to parallelize your experiments,
    as otherwise it may take months to run the experiments.
    The benchmark currently only supports native parallelization in `aws` mode
    (by using the `--parallel` parameter), but using the `--task` and `--fold` parameters 
    it is easy to generate scripts that invoke individual jobs on e.g., a SLURM cluster.
    When you run in any parallelized fashion, it is advised to run each process on
    separate hardware to ensure experiments can not interfere with each other.
