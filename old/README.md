# OpenML AutoML Benchmark

The OpenML AutoML Benchmark provides a framework for evaluating and comparing open-source AutoML systems.  The system is *extensible* because you can [add your own](https://github.com/openml/automlbenchmark/blob/master/docs/extending.md) AutoML frameworks and datasets. For a thorough explanation of the benchmark, and evaluation of results, you can read our [paper](https://openml.github.io/automlbenchmark/paper.html) which was accepted at the [2019 ICML AutoML Workshop](https://sites.google.com/view/automl2019icml/).

_**NOTE:**_ _This benchmarking framework currently features binary and multiclass classification; extending to regression is a work in progress.  Please file an issue with any concerns/questions._

  * [Installation](#installation)
     * [Pre-requisites](#pre-requisites)
     * [Setup](#setup)
  * [Quickstart](#quickstart)
  * [Running benchmarks](#running-benchmarks)
     * [In Docker image](#in-docker-image)
     * [In local environment](#in-local-environment)
     * [On AWS](#on-aws)
     * [Output](#output)
  * [Advanced configuration](#advanced-configuration)
  * [Issues](#issues)
  * [Frequently Asked Questions](#frequently-asked-questions)
      
Automatic Machine Learning (AutoML) systems automatically build machine learning pipelines or neural architectures in a data-driven, objective, and automatic way. They automate a lot of drudge work in designing machine learning systems, so that better systems can be developed, faster. However, AutoML research is also slowed down by two factors:

* We currently lack standardized, easily-accessible benchmarking suites of tasks (datasets) that are curated to reflect important problem domains, practical to use, and sufficiently challenging to support a rigorous analysis of performance results. 

* Subtle differences in the problem definition, such as the design of the hyperparameter search space or the way time budgets are defined, can drastically alter a task’s difficulty. This issue makes it difficult to reproduce published research and compare results from different papers.

This toolkit aims to address these problems by setting up standardized environments for in-depth experimentation with a wide range of AutoML systems.

Documentation: <https://openml.github.io/automlbenchmark/>

### Features:
* Curated suites of [benchmarking datasets](https://openml.github.io/automlbenchmark/benchmark_datasets.html) from [OpenML](https://www.openml.org/s/218/data).
* Includes code to benchmark a number of [popular AutoML systems](https://openml.github.io/automlbenchmark/automl_overview.html) on regression and classification tasks.
* [New AutoML systems can be added](./HOWTO.md#add-an-automl-framework)
* Experiments can be run in Docker or Singularity containers
* Execute experiments locally or on AWS (see below)


## Installation
### Pre-requisites
To run the benchmarks, you will need:
* Python 3.9+.
* PIP3: ensure you have a recent version. If necessary, upgrade your pip using `python -m pip install -U pip`.
* The Python libraries listed in [requirements.txt](../requirements.txt): it is strongly recommended to first create a [Python virtual environment](https://docs.python.org/3/library/venv.html#venv-def) (cf. also [Pyenv](https://github.com/pyenv/pyenv): quick install using `curl https://pyenv.run | bash` or `brew install pyenv`) and work in it if you don't want to mess up your global Python environment.
* [Docker](https://docs.docker.com/install/), if you plan to run the benchmarks in a container.

### Setup
Clone the repo (in development environment, you should of course remove the `--depth 1` argument):
```bash
git clone https://github.com/openml/automlbenchmark.git --branch stable --depth 1
cd automlbenchmark
```
Optional: create a Python3 virtual environment.

- _**NOTE**: we don't recommend creating your virtual environment with `virtualenv` library here as the application may create additional virtual environments for some frameworks to run in isolation._
_Those virtual environments are created internally using `python -m venv` and we encountered issues with `pip` when `venv` is used on top of a `virtualenv` environment._
_Therefore, we rather suggest one of the method below:_ 

using venv on Linux/macOS:
```bash
python3 -m venv ./venv
source venv/bin/activate
# remember to call `deactivate` once you're done using the application
```
using venv on Windows:
```bash
python3 -m venv ./venv
venv\Scripts\activate
# remember to call `venv\Scripts\deactivate` once you're done using the application
```

or using pyenv:
```bash
pyenv install {python_version: 3.9.16}
pyenv virtualenv ve-automl
pyenv local ve-automl
```
Then pip install the dependencies:

```bash
python -m pip install -r requirements.txt
```

- _**NOTE**: in case of issues when installing Python requirements, you may want to try the following:_
    - _on some platforms, we need to ensure that requirements are installed sequentially:_ `xargs -L 1 python -m pip install < requirements.txt`.
    - _enforce the `python -m pip` version above in your virtualenv:_ `python -m pip install --upgrade pip==19.3.1`.

## Quickstart
To run a benchmark call the `runbenchmark.py` script with at least the following arguments:

1. The AutoML framework that should be evaluated, see [frameworks.yaml](../resources/frameworks.yaml) for supported frameworks. If you want to add a framework see [HOWTO](./HOWTO.md#add-an-automl-framework).
2. The benchmark suite to run should be one implemented in [benchmarks folder](../resources/benchmarks), or an OpenML study or task (formatted as `openml/s/X` or `openml/t/Y` respectively).
3. (Optional) The constraints applied to the benchmark as defined by default in [constraints.yaml](../resources/constraints.yaml). Default constraint is `test` (2 folds for 10 min each).
4. (Optional) If the benchmark should be run `local` (default, tested on Linux and macOS only), in a `docker` container or on `aws` using multiple ec2 instances.

Examples:
```bash
python3 runbenchmark.py 
python3 runbenchmark.py constantpredictor
python3 runbenchmark.py tpot test
python3 runbenchmark.py autosklearn openml/t/59 -m docker
python3 runbenchmark.py h2oautoml validation 1h4c -m aws
python3 runbenchmark.py autogluon:latest validation
python3 runbenchmark.py tpot:2020Q2
```

For the complete list of supported arguments, run:
```bash
python3 runbenchmark.py --help
```

```text
usage: runbenchmark.py [-h] [-m {local,aws,docker,singularity}]
                       [-t [task_id [task_id ...]]]
                       [-f [fold_num ...]] [-i input_dir]
                       [-o output_dir] [-u user_dir] [-p parallel_jobs]
                       [-s {auto,skip,force,only}] [-k [true|false]]
                       [-e] [--logging LOGGING]
                       [--openml-run-tag OPENML_RUN_TAG]
                       framework [benchmark] [constraint]

positional arguments:
  framework             The framework to evaluate as defined by default in resources/frameworks.yaml.
                        To use a labelled framework (i.e. a framework defined in resources/frameworks-{label}.yaml),
                        use the syntax {framework}:{label}.
  benchmark             The benchmark type to run as defined by default in resources/benchmarks/{benchmark}.yaml,
                        a path to a benchmark description file, or an openml suite or task.
                        OpenML references should be formatted as 'openml/s/X' and 'openml/t/Y',
                        for studies and tasks respectively. Use 'test.openml/s/X' for the 
                        OpenML test server.
                        (default: 'test')
  constraint            The constraint definition to use as defined by default in resources/constraints.yaml.
                        (default: 'test')

optional arguments:
  -h, --help            show this help message and exit
  -m {local,aws,docker,singularity}, --mode {local,aws,docker,singularity}
                        The mode that specifies how/where the benchmark tasks will be running.
                        (default: 'local')
  -t [task_id ...], --task [task_id ...]
                        The specific task name (as defined in the benchmark file) to run.
                        When an OpenML reference is used as benchmark, the dataset name should be used instead.
                        If not provided, then all tasks from the benchmark will be run.
  -f [fold_num ...], --fold [fold_num ...]
                        If task is provided, the specific fold(s) to run.
                        If fold is not provided, then all folds from the task definition will be run.
  -i input_dir, --indir input_dir
                        Folder from where the datasets are loaded by default.
                        (default: '~/.openml')
  -o output_dir, --outdir output_dir
                        Folder where all the outputs should be written.(default: './results')
  -u user_dir, --userdir user_dir
                        Folder where all the customizations are stored.(default: '~/.config/automlbenchmark')
  -p parallel_jobs, --parallel parallel_jobs
                        The number of jobs (i.e. tasks or folds) that can run in parallel.
                        A hard limit is defined by property `job_scheduler.max_parallel_jobs`
                         in `resources/config.yaml`.
                        Override this limit in your custom `config.yaml` file if needed.
                        Supported only in aws mode or container mode (docker, singularity).
                        (default: 1)
  -s {auto,skip,force,only}, --setup {auto,skip,force,only}
                        Framework/platform setup mode. Available values are:
                        • auto: setup is executed only if strictly necessary.
                        • skip: setup is skipped.
                        • force: setup is always executed before the benchmark.
                        • only: only setup is executed (no benchmark).
                        (default: 'auto')
  -k [true|false], --keep-scores [true|false]
                        Set to true (default) to save/add scores in output directory.
  -e, --exit-on-error   If set, terminates on the first task that does not complete with a model.
  --logging LOGGING     Set the log levels for the 3 available loggers:
                        • console
                        • app: for the log file including only logs from amlb (.log extension).
                        • root: for the log file including logs from libraries (.full.log extension).
                        Accepted values for each logger are: notset, debug, info, warning, error, fatal, critical.
                        Examples:
                          --logging=info (applies the same level to all loggers)
                          --logging=root:debug (keeps defaults for non-specified loggers)
                          --logging=console:warning,app:info
                        (default: 'console:info,app:debug,root:info')
  --openml-run-tag OPENML_RUN_TAG
                        Tag that will be saved in metadata and OpenML runs created during upload, must match '([a-zA-Z0-9_\-\.])+'.
```

The script will produce output that records task metadata and the result.
The result is the score on the test set, where the score is a specific model performance metric (e.g. "AUC") defined by the benchmark.
```text
   task  framework  fold    result   mode   version                  utc       acc       auc   logloss
0  iris  H2OAutoML     0  1.000000  local  3.22.0.5  2019-01-21T15:19:07  1.000000       NaN  0.023511
1  iris  H2OAutoML     1  1.000000  local  3.22.0.5  2019-01-21T15:20:12  1.000000       NaN  0.091685
2   kc2  H2OAutoML     0  0.811321  local  3.22.0.5  2019-01-21T15:21:11  0.811321  0.859307       NaN
3   kc2  H2OAutoML     1  0.886792  local  3.22.0.5  2019-01-21T15:22:12  0.886792  0.888528       NaN
```

## Running benchmarks
The `automlbenchmark` app currently allows running benchmarks in various environments:
* in a docker container (running locally or on multiple AWS instances).
* completely locally, if the framework is supported on the local system.
* on AWS, possibly distributing the tasks to multiple EC2 instances, each of them running the benchmark either locally or in a docker container.

### In Docker image
The [Docker] image is automatically built before running the benchmark if it doesn't already exist locally or in a public repository (by default in <https://hub.docker.com/orgs/automlbenchmark/repositories>).
Especially, without docker image, the application will need to download and install all the dependencies when building the image, so this may take some time.

The generated image is usually named `automlbenchmark/{framework}:{tag}`, but this is customizable per framework: cf. `resources/frameworks.yaml` and [HOWTO](HOWTO.md#framework-definition) for details.

For example, this will build a Docker image for the `RandomForest` framework and then immediately start a container to run the `validation` benchmark, using all folds, allocating 1h and 4 cores for each task:
```bash
python3 runbenchmark.py RandomForest validation 1h4c -m docker
```

If the corresponding image already exists locally and you want it to be rebuilt before running the benchmark, then the setup needs to be forced:
```bash
python3 runbenchmark.py {framework} {benchmark} {constraint} -m docker -s force
```

The image can also be built without running any benchmark:
```bash
python3 runbenchmark.py {framework} -m docker -s only
```

In rare cases, mainly for development, you may want to specify the docker image:
```bash
python3 runbenchmark.py {framework} {benchmark} {constraint} -m docker -Xdocker.image={image}
```

### In local environment
If docker allows portability, it is still possible to run the benchmarks locally without container on some environments (currently Linux, and macOS for most frameworks).

A minimal example would be to run the test benchmarks with a random forest:
```bash
python3 runbenchmark.py RandomForest test
```

The majority of frameworks though require a `setup` step before being able to run a benchmark. Please note that this step may take some time depending on the framework.
This setup is executed by default on first run of the framework, but in this case, it is not guaranteed that the benchmark run following immediately will manage to complete successfully (for most frameworks though, it does).

In case of error, just run the benchmark one more time.

If it still fails, you may need to rerun the setup step manually:
```bash
python3 runbenchmark.py {framework} -s only
```
You can then run the benchmarks as many times as you wish.

When testing a framework or a new dataset, you may want to run only a single task and a specific fold, for example:
```bash
python3 runbenchmark.py TPOT validation -t bioresponse -f 0
```
