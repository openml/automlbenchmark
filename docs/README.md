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
* Includes code to benchmark a number of [popular AutoML systems](https://openml.github.io/automlbenchmark/automl_overview.html)
* [New AutoML systems can be added](./HOWTO.md#add-an-automl-framework)
* Experiments can be run in Docker or Singularity containers
* Execute experiments locally or on AWS (see below)

### Roadmap: 
* Add a regression benchmark to the framework.
* More benchmark datasets (especially bigger datasets).
* Automatic sharing of benchmarking results on OpenML.


## Installation
### Pre-requisites
To run the benchmarks, you will need:
* Python 3.6+.
* PIP3: ensure you have a recent version. If necessary, upgrade your pip using `pip3 install --upgrade pip`.
* The Python libraries listed in [requirements.txt](../requirements.txt): it is strongly recommended to first create a [Python virtual environment](https://docs.python.org/3/library/venv.html#venv-def) (cf. also [Pyenv](https://github.com/pyenv/pyenv): quick install using `curl https://pyenv.run | bash` or `brew install pyenv`) and work in it if you don't want to mess up your global Python environment.
* [Docker](https://docs.docker.com/install/), if you plan to run the benchmarks in a container.

### Setup
Clone the repo:
```bash
git clone https://github.com/openml/automlbenchmark.git --branch stable --depth 1
cd automlbenchmark
```
Optional: create a Python3 virtual environment.

- _**NOTE**: we don't recommend to create your virtual environment with `virtualenv` library here as the application may create additional virtual environments for some frameworks to run in isolation._
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
pyenv install {python_version: 3.7.4}
pyenv virtualenv ve-automl
pyenv local ve-automl
```
Then pip install the dependencies:

```bash
pip3 install -r requirements.txt
```

- _**NOTE**: in case of issues when installing Python requirements, you may want to try the following:_
    - _on some platforms, we need to ensure that requirements are installed sequentially:_ `xargs -L 1 pip install < requirements.txt`.
    - _enforce the `pip3` version above in your virtualenv:_ `pip3 install --upgrade pip==19.3.1`.


## Quickstart
To run a benchmark call the `runbenchmark.py` script with at least the following arguments:

1. The AutoML framework that should be evaluated, see [frameworks.yaml](../resources/frameworks.yaml) for supported frameworks. If you want to add a framework see [HOWTO](./HOWTO.md#add-an-automl-framework).
2. The benchmark suite to run should be one implemented in [benchmarks folder](../resources/benchmarks), or an OpenML study or task (formatted as `openml/s/X` or `openml/t/Y` respectively).
3. (Optional) The constraints applied to the benchmark as defined by default in [constraints.yaml](../resources/constraints.yaml). Default constraint is `test` (1 single fold for 10 min).
4. (Optional) If the benchmark should be run `local` (default, tested on Linux and macOS only), in a `docker` container or on `aws` using multiple ec2 instances.

Examples:
```bash
python3 runbenchmark.py 
python3 runbenchmark.py constantpredictor
python3 runbenchmark.py tpot test
python3 runbenchmark.py autosklearn openml/t/59 -m docker
python3 runbenchmark.py h2oautoml validation 1h4c -m aws
```

For the complete list of supported arguments, run:
```bash
python3 runbenchmark.py --help
```

```text
usage: runbenchmark.py [-h] [-m {local,docker,aws}]
                       [-t [task_id [task_id ...]]]
                       [-f [fold_num [fold_num ...]]] [-i input_dir]
                       [-o output_dir] [-u user_dir] [-p parallel_jobs]
                       [-s {auto,skip,force,only}] [-k [true|false]]
                       framework [benchmark] [constraint]

positional arguments:
  framework             The framework to evaluate as defined by default in
                        resources/frameworks.yaml.
  benchmark             The benchmark type to run as defined by default in resources/benchmarks/{benchmark}.yaml, 
                        a path to a benchmark description file, or an openml suite or task. 
                        OpenML references should be formatted as 'openml/s/X'  and 'openml/t/Y', 
                        for studies and tasks respectively. Defaults to `test`.
  constraint            The constraint definition to use as defined by default in
                        resources/constraints.yaml. Defaults to `test`.

optional arguments:
  -h, --help            show this help message and exit
  -m {local,docker,aws}, --mode {local,docker,aws}
                        The mode that specifies how/where the benchmark tasks
                        will be running. Defaults to local.
  -t [task_id [task_id ...]], --task [task_id [task_id ...]]
                        The specific task name (as defined in the benchmark
                        file) to run. When an OpenML reference is used as benchmark, 
                        the dataset name should be used instead. If not provided, 
                        then all tasks from the benchmark will be run.
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
  -u user_dir, --userdir user_dir
                        Folder where all the customizations are stored.
                        Defaults to `user_dir` as defined in
                        resources/config.yaml
  -p parallel_jobs, --parallel parallel_jobs
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

### On AWS
To run a benchmark on AWS you additionally need to have a configured AWS account.
The application is using the [boto3] Python package to exchange files through S3 and create EC2 instances.

 If this is your first time setting up your AWS account on the machine that will run the `automlbenchmark` app, you can use the [AWS CLI](http://aws.amazon.com/cli/) tool and run:
 ```bash
 aws configure
 ```
You will need your AWS Access Key ID, AWS Secret Access Key, and pick a default [EC2 region](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-regions-availability-zones.html#concepts-available-regions).

- _**NOTE:** Currently the AMI is only configured for the following regions so you'll have to set your default region as one of these_:
  - us-east-1
  - us-west-1
  - eu-west-1
  - eu-central-1
  
On first use, it is recommended to simply copy the `config.yaml` from [examples/aws] to your user `~/.config/automlbenchmark` folder (or merge it if you already have a `config.yaml` in this user folder) and follow the instructions in that file.

To run a test to see if the benchmark framework is working on AWS, do the following:
```bash
python3 runbenchmark.py constantpredictor test -m aws
```
This will create and start an EC2 instance for each benchmark job and run the 4 jobs (2 OpenML tasks * 2 folds) from the `test` benchmark sequentially, each job running for 1mn in this case (excluding setup time for the EC2 instances).

For longer benchmarks, you'll probably want to run multiple jobs in parallel and distribute the work to several EC2 instances, for example:
```bash
python3 runbenchmark.py AUTOWEKA validation 1h4c -m aws -p 4
```
will keep 4 EC2 instances running, monitor them in a dedicated thread, and finally collect all outputs from s3.

- _**NOTE**: each EC2 instance is provided with a time limit at startup to ensure that in any case, the instance is stopped even if there is an issue when running the benchmark task. In this case the instance is stopped, not terminated, and we can therefore inspect the machine manually (ideally after resetting its UserData field to avoid re-triggering the benchmark on the next startup)._

The console output is still showing the instances starting, outputs the progress and then the results for each dataset/fold combination:
```text
Running `H2OAutoML_nightly` on `validation` benchmarks in `aws` mode
Loading frameworks definitions from ['/Users/me/repos/automlbenchmark/resources/frameworks.yaml'].
Loading benchmark definitions from /Users/me/repos/automlbenchmark/resources/benchmarks/validationt.yaml.
Uploading `/Users/me/repos/automlbenchmark/resources/benchmarks/validation.yaml` to `ec2/input/validation.yaml` on s3 bucket automl-benchmark.
...
Starting new EC2 instance with params: H2OAutoML_nightly /s3bucket/input/validation.yaml -t micro-mass -f 0
Started EC2 instance i-0cd081efc97c3bf6f 
[2019-01-22T11:51:32] checking job aws_validation_micro-mass_0_H2OAutoML_nightly on instance i-0cd081efc97c3bf6f: pending 
Starting new EC2 instance with params: H2OAutoML_nightly /s3bucket/input/validation.yaml -t micro-mass -f 1
Started EC2 instance i-0251c1655e286897c 
...
[2019-01-22T12:00:32] checking job aws_validation_micro-mass_1_H2OAutoML_nightly on instance i-0251c1655e286897c: running
[2019-01-22T12:00:33] checking job aws_validation_micro-mass_0_H2OAutoML_nightly on instance i-0cd081efc97c3bf6f: running
[2019-01-22T12:00:48] checking job aws_validation_micro-mass_1_H2OAutoML_nightly on instance i-0251c1655e286897c: running
[2019-01-22T12:00:48] checking job aws_validation_micro-mass_0_H2OAutoML_nightly on instance i-0cd081efc97c3bf6f: running
...
[  731.511738] cloud-init[1521]: Predictions saved to /s3bucket/output/predictions/h2oautoml_nightly_micro-mass_0.csv
[  731.512132] cloud-init[1521]: H2O session _sid_96e7 closed.
[  731.512506] cloud-init[1521]: Loading predictions from /s3bucket/output/predictions/h2oautoml_nightly_micro-mass_0.csv
[  731.512890] cloud-init[1521]: Metric scores: {'framework': 'H2OAutoML_nightly', 'version': 'nightly', 'task': 'micro-mass', 'fold': 0, 'mode': 'local', 'utc': '2019-01-22T12:00:02', 'logloss': 0.6498889633819804, 'acc': 0.8793103448275862, 'result': 0.6498889633819804}
[  731.513275] cloud-init[1521]: Job local_micro-mass_0_H2OAutoML_nightly executed in 608.534 seconds
[  731.513662] cloud-init[1521]: All jobs executed in 608.534 seconds
[  731.514089] cloud-init[1521]: Scores saved to /s3bucket/output/scores/H2OAutoML_nightly_task_micro-mass.csv
[  731.514542] cloud-init[1521]: Loaded scores from /s3bucket/output/scores/results.csv
[  731.515006] cloud-init[1521]: Scores saved to /s3bucket/output/scores/results.csv
[  731.515357] cloud-init[1521]: Summing up scores for current run:
[  731.515782] cloud-init[1521]:          task          framework    ...         acc   logloss
[  731.516228] cloud-init[1521]: 0  micro-mass  H2OAutoML_nightly    ...     0.87931  0.649889
[  731.516671] cloud-init[1521]: [1 rows x 9 columns]
...
EC2 instance i-0cd081efc97c3bf6f is stopped
Job aws_validation_micro-mass_0_H2OAutoML_nightly executed in 819.305 seconds
[2019-01-22T12:01:34] checking job aws_validation_micro-mass_1_H2OAutoML_nightly on instance i-0251c1655e286897c: running
[2019-01-22T12:01:49] checking job aws_validation_micro-mass_1_H2OAutoML_nightly on instance i-0251c1655e286897c: running
EC2 instance i-0251c1655e286897c is stopping
Job aws_validation_micro-mass_1_H2OAutoML_nightly executed in 818.463 seconds
...
Terminating EC2 instances i-0251c1655e286897c
Terminated EC2 instances i-0251c1655e286897c with response {'TerminatingInstances': [{'CurrentState': {'Code': 32, 'Name': 'shutting-down'}, 'InstanceId': 'i-0251c1655e286897c', 'PreviousState': {'Code': 64, 'Name': 'stopping'}}], 'ResponseMetadata': {'RequestId': 'd09eeb0c-7a58-4cde-8f8b-2308a371a801', 'HTTPStatusCode': 200, 'HTTPHeaders': {'content-type': 'text/xml;charset=UTF-8', 'transfer-encoding': 'chunked', 'vary': 'Accept-Encoding', 'date': 'Tue, 22 Jan 2019 12:01:53 GMT', 'server': 'AmazonEC2'}, 'RetryAttempts': 0}}
Instance i-0251c1655e286897c state: shutting-down
All jobs executed in 2376.891 seconds
Deleting uploaded resources `['ec2/input/validation.yaml', 'ec2/input/config.yaml', 'ec2/input/frameworks.yaml']` from s3 bucket automl-benchmark.
```

### Output
By default, a benchmark run creates the following subdirectories and files in the output directory (by default a subdirectory of `./results` with unique name identifying the benchmark run):
* `scores`: this subdirectory contains
    * `results.csv`: a global scoreboard, keeping scores from all benchmark runs. 
       For safety reasons, this file is automatically backed up to `scores/backup/results_{currentdate}.csv` by the application before any modification. 
    * individual score files keeping scores for each framework+benchmark combination (not backed up). 
* `predictions`, this subdirectory contains the last predictions in a standardized format made by each framework-dataset combination.
  Those last predictions are systematically backed up with current data to `predictions/backup` subdirectory before a new prediction is written.
* `logs`: this subdirectory contains logs produced by the `automlbenchmark` app, including when it's been run in Docker container or on AWS.


## Advanced configuration
If you need to create your own benchmark, add a framework, create a plugin for a proprietary framework, or simply want to use some advanced options (e.g. run some frameworks with non-default parameters), see the [HOWTO].

## Issues
If you face any issue, please first have a look at the [Troubleshooting guide] and check the [existing issues](https://github.com/openml/automlbenchmark/issues).
Any new issue should also be reported there.


[HOWTO]: ./HOWTO.md
[Troubleshooting guide]: ./HOWTO.md#troubleshooting-guide
[examples/aws]: ../examples/aws/config.yaml

[Docker]: https://docs.docker.com/
[boto3]: https://boto3.readthedocs.io/


## Frequently Asked Questions

**When will results be updated, also for the new/updated frameworks?**

We don't perform a benchmark evaluation for each new package or update.
Due to budget constraints, we can only do a limited number of evaluations.
The next full evaluation will be performed before the end of the year 2020.
We hope to find funding to guarantee regular evaluations.

---
**(When) will you add framework X?**

We are currently not focused on integrating additional AutoML systems.
However, we process any pull requests that add frameworks and will assist with the integration.
The best way to make sure framework X gets included is to start with the integration yourself or encourage the package authors to do so (for technical details see [HOWTO]).

It is also possible to open a Github issue indicating the framework you would like added.
Please use a clear title (e.g. "Add framework: X") and provide some relevant information (e.g. a link to the documentation).
This helps us keep track of which frameworks people are interested in seeing included.
