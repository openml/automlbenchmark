# AutoML Benchmarking

_**NOTE:**_ _This benchmarking framework is a WORK IN PROGRESS.  Check back later for the completed benchmark suite.  Please file an issue with any concerns/questions._

Automatic Machine Learning (AutoML) systems automatically build machine learning pipelines or neural architectures in a data-driven, objective, and automatic way. They automate a lot of drudge work in designing machine learning systems, so that better systems can be developed, faster. However, AutoML research is also slowed down by two factors:

* We currently lack standardized, easily-accessible benchmarking suites of tasks (datasets) that are curated to reflect important problem domains, practical to use, and sufficiently challenging to support a rigorous analysis of performance results. 

* Subtle differences in the problem definition, such as the design of the hyperparameter search space or the way time budgets are defined, can drastically alter a task’s difficulty. This issue makes it difficult to reproduce published research and compare results from different papers.

This toolkit aims to address these problems by setting up standardized environments for in-depth experimentation with a wide range of AutoML systems.

Documentation: https://openml.github.io/automlbenchmark/

### Features:
* Curated suites of benchmarking datasets from OpenML (TODO: add study link)
* Includes a [wide range of AutoML systems](https://openml.github.io/automlbenchmark/automl_overview.html)
* [New AutoML systems can be added](https://github.com/openml/automlbenchmark/tree/master/docker) as Docker images
* Execute experiments locally or on AWS (see below)

Future plans:  
* Automatic sharing of benchmarkling results on OpenML.
* Allow tuning of the AutoML systems (hyper-hyperparameters), beyond their default settings.
* More benchmark datasets, and datasets of other types (e.g. regression).

## Installation
To run the benchmarks, you will need:
* Python 3.5+ (TODO: verify work necessary to support Py2 and older versions of Py3).
* the Python libraries listed in [requirements.txt](requirements.txt): it is strongly recommended to first create a [Python virtual environment](https://docs.python.org/3/library/venv.html#venv-def) (cf. also [Pyenv](https://github.com/pyenv/pyenv)) and work in it if you don't want to mess up your global Python environment.
* the [OpenML](https://github.com/openml/openml-python). (The Python requirements currently fails installing if `openml` is included in `requirements.txt` when `numpy` is not already installed).
* [Docker](https://docs.docker.com/install/), if you plan to run the benchmarks in a container.

```bash
git clone https://github.com/openml/automlbenchmark.git
cd automlbenchmark
pip3 install -r requirements.txt
pip3 install openml
```

## Quickstart
To run a benchmark call the `runbenchmark.py` script with at least the following arguments:

1. The AutoML framework that should be evaluated, see [frameworks.yaml](resources/frameworks.yaml) for supported frameworks. If you want to add a framework see [here](docker/readme.md).
2. The benchmark suite to run. Should be one implemented in [benchmarks folder](resources/benchmarks).
3. (Optional) If the benchmark should be run `local` (default, tested on Linux and macOS only), in a `docker` container or on `aws` using multiple ec2 instances.

Examples:
```bash
python runbenchmark.py autosklearn test -m docker

python runbenchmark.py h2oautoml validation -m aws
```

For the complete list of supported arguments, run:
```bash
python runbenchmark.py --help
```

```text
usage: runbenchmark.py [-h] [-m {local,docker,aws}] [-t task_id]
                       [-f [fold_num [fold_num ...]]] [-i input_dir]
                       [-o output_dir] [-p jobs_count]
                       [-s {auto,skip,force,only}] [-k [true|false]]
                       framework [benchmark]

positional arguments:
  framework             The framework to evaluate as defined by default in
                        resources/frameworks.yaml.
  benchmark             The benchmark type to run as defined by default in
                        resources/benchmarks/{benchmark}.yaml or the path to a
                        benchmark description file. Defaults to `test`.

optional arguments:
  -h, --help            show this help message and exit
  -m {local,docker,aws}, --mode {local,docker,aws}
                        The mode that specifies how/where the benchmark tasks
                        will be running. Defaults to local.
  -t task_id, --task task_id
                        The specific task name (as defined in the benchmark
                        file) to run. If not provided, then all tasks from the
                        benchmark will be run.
  -f [fold_num [fold_num ...]], --fold [fold_num [fold_num ...]]
                        If task is provided, the specific fold(s) to run. If
                        fold is not provided, then all folds from the task
                        definition will be run.
  -i input_dir, --indir input_dir
                        Folder where datasets are loaded by default. Defaults
                        to `input_dir` as defined in resources/config.yaml
  -o output_dir, --outdir output_dir
                        Folder where all the outputs should be written.
                        Defaults to `output_dir` as defined in
                        resources/config.yaml
  -p jobs_count, --parallel jobs_count
                        The number of jobs (i.e. tasks or folds) that can run
                        in parallel. Defaults to 1. Currently supported only
                        in docker and aws mode.
  -s {auto,skip,force,only}, --setup {auto,skip,force,only}
                        Framework/platform setup mode. Defaults to auto.
                        •auto: setup is executed only if strictly necessary.
                        •skip: setup is skipped. •force: setup is always
                        executed before the benchmark. •only: only setup is
                        executed (no benchmark).
  -k [true|false], --keep-scores [true|false]
                        Set to true [default] to save/add scores in output
                        directory.
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
The Docker image is automatically built before running the benchmark if it doesn't already exist locally or in a public repository.
Especially, it will need to download and install all the dependencies when building the image, so this may take some time.

The generated image is usually named `automlbenchmark/{framework}:{tag}`, but this is customizable per framework: cf. `resources/frameworks.yaml` for details.

For example, this will build a Docker image for the `RandomForest` framework and then immediately start a container to run the `validation` benchmark:
```bash
python runbenchmark.py RandomForest validation -m docker
```

If the corresponding image already exists locally and you want it to be rebuilt before running the benchmark, then the setup needs to be forced:
```bash
python runbenchmark.py RandomForest validation -m docker -s force
```

The image can also be built without running any benchmark:
```bash
python runbenchmark.py {framework} -m docker -s only
```

### In local environment
If docker allows portability, it is still possible to run the benchmarks locally without container on some environments (currently Linux, and macOS for most frameworks).

First, the framework needs to be set up locally, which may also take some time:
```bash
python runbenchmark.py {framework} -s only
```
You can then run the benchmarks as many times as you wish.
A minimal example would be to run the test benchmarks with a random forest:
```bash
python runbenchmark.py RandomForest test
```

When testing a framework or a new dataset, you may want to run only a single task and a specific fold:
```bash
python runbenchmark.py TPOT validation -t bioresponse -f 0
```

### On AWS
To run a benchmark on AWS you additionally need to have a configured AWS account.
The application is using the [boto3](https://boto3.readthedocs.io/) Python package to exchange files through S3 and create EC2 instances.

 If this is your first time setting up your AWS account on the machine that will run the `automlbenchmark` app, you can use the [AWS CLI](http://aws.amazon.com/cli/) tool and run:
 ```bash
 aws configure
 ```
You will need your AWS Access Key ID, AWS Secret Access Key, a default [EC2 region](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-regions-availability-zones.html#concepts-available-regions) and select "text" as the default output format.

- _**NOTE:** Currently the AMI is only configured for the following regions so you'll have to set your default region as one of these_:
  - us-east-1
  - us-west-1
  - eu-west-1
  - eu-central-1

To run a test to see if the benchmark framework is working on AWS, do the following:
```bash
python benchmark.py RandomForest test -m aws
```
This will create and start an EC2 instance for each benchmark job and run the 4 jobs (2 OpenML tasks * 2 folds) from the `test` benchmark sequentially, each job running for 1mn in this case (excluding setup time for the EC2 instances).

For longer benchmarks, you'll probably want to run multiple jobs in parallel and distribute the work to several EC2 instances, for example:
```bash
python benchmark.py AUTOWEKA validation -m aws -p 4
```
will keep 4 EC2 instances running, monitor them in a dedicated thread, and finally collect all outputs from s3.

**NOTE**: each EC2 instance is provided with a time limit at startup, to ensure that in any case, the instance is stopped even if there is an issue when running the benchmark task. In this case the instance is stopped, not terminated, and we can therefore inspect the machine manually (ideally after resetting its UserData field to avoid re-triggering the benchmark on the next startup).

The console output is still showing the instances starting, outputs the progress and then the results for each dataset/fold combination:
```text
Console output example HERE
```

### Output
By default, a benchmark run creates the following subdirectories and files in the output directory (by default `./results`):
* `scores`: this subdirectory contains
    * `results.csv`: a global scoreboard, keeping scores from all benchmark runs. 
       For safety reasons, this file is automatically backed up to `scores/backup/results_{currentdate}.csv` by the application before any modification. 
    * individual score files keeping scores for each framework+benchmark combination (not backed up). 
* `predictions`, this subdirectory contains the last predictions in a standardized format made by each framework-dataset combination.
  Those last predictions are systematically backed up with current data to `predictions/backup` subdirectory before a new prediction is written.
* `logs`: this subdirectory contains logs produced by the `automlbenchmark` app, including when it's been run in Docker container or on AWS.


## Advanced configuration

### Running a custom benchmark


### Running a custom framework

### Benchmark constraints
##### Time limits
##### Memory
##### CPUs

### AWS config
