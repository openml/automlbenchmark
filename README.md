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

## Quickstart

To run a benchmark call the `runbenchmark.py` file at least arguments:

1. The AutoML framework that should be evaluated, see [frameworks.yaml](resources/frameworks.yaml) for supported frameworks. If you want to add a framework see [here](docker/readme.md).
2. The benchmark suite to run. Should be one implemented in [benchmarks folder](resources/benchmarks).
3. (Optional) If the benchmark should be run `local` (default, tested on Linux and macOS only), in a `docker` container or on `aws` using multiple ec2 instances.

Examples:
```bash
python runbenchmark.py autosklearn validation

python runbenchmark.py h2oautoml validation -m aws
```

For the complete list of supported arguments, run:
```bash
python runbenchmark.py --help
```


## Installation

To run the benchmarks, you will need:
* Python 3.5+ (TODO: verify work necessary to support Py2 and older versions of Py3).
* the Python libraries listed in [requirements.txt](requirements.txt).
* the [OpenML](https://github.com/openml/openml-python) Python client (currently fails installing if included in requirements.txt when numpy is not already installed).
* [Docker](https://docs.docker.com/install/), if you plan to run the benchmarks in a container.

```
git clone https://github.com/openml/automlbenchmark.git
cd automlbenchmark
pip3 install -r requirements.txt
pip3 install openml
```

### Generate Docker Images

The Docker image is automatically built before running the benchmark if it doesn't exist locally or in a public repository.
The generated image is usually named: automlbenchmark/{framework}:{tag}

```bash
python runbenchmark.py RandomForest validation -m docker
```

To build the image without running any benchmark:
```bash
python runbenchmark.py TPOT -m docker -s only
```


### Run the benchmark locally

A minimal example would be to run the test benchmarks with a random forest:

```
python runbenchmark.py RandomForest test local
```
The first time you execute the benchmark, it will download all the dependencies to install in the Docker image, so that will take some time.

The script will produce output that records the OpenML Task ID, the fold index the result.  The result is the score on the test set, where the score is a specific model performance metric (e.g. "AUC") defined by the benchmark.

```
  benchmark_id  fold    result
0       test_1     0  0.933333
1       test_1     1  1.000000
2       test_2     0  0.811321
3       test_2     1  0.849057
```


### Run the benchmark on AWS

To run a benchmark on AWS you additionally need to:

- Have the `boto3` Python package installed and [configured](https://boto3.readthedocs.io/en/latest/guide/quickstart.html#configuration) on your machine to have access to your AWS account credentials. If this is your first time setting up **boto3**, you can use the `aws configure` tool and will need your AWS Access Key ID, AWS Secret Access Key, a default [EC2 region](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-regions-availability-zones.html#concepts-available-regions) and you can select "text" as the default output format.
- _**NOTE:** Currently the AMI is only in the following regions so you'll have to set your default region as one of these_:
  - us-east-1
  - us-west-1
  - eu-west-1

To run a test  to see if the benchmark framework is working on AWS, do the following:

```
python benchmark.py RandomForest test aws
```

The output shows the instances starting, outputs the progress and then the results for each dataset/fold combination:

```
Grouping 4 jobs in 1 chunk(s) of 4 parallel jobs
---- Chunk 1/1 ----
Created 4 jobs
Starting instances
Instance pending
Instance pending
Instance pending
Instanceending
[00:00:10] - 0/4 jobs done
[00:00:25] - 0/4 jobs done
[00:00:40] - 0/4 jobs done
[00:00:56] - 0/4 jobs done
[00:01:12] - 4/4 jobs done
Chunk 1 done, terminating Instances:
Termination successful
Termination successful
Termination successful
Termination successful
  benchmark_id  fold              result
0       test_1     0  0.9333333333333333
1       test_1     1                 1.0
2       test_2     0  0.8679245283018868
3       test_2     1  0.8679245283018868
```

