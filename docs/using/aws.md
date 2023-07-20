# AWS

The AutoML benchmark supports running experiments on [AWS EC2](https://aws.amazon.com/ec2/).

!!! danger "AMLB does not limit expenses!"

    The AWS integration lets your easily conduct massively parallel evaluations.
    The AutoML Benchmark does not in any way restrict the _total_ costs you can make on AWS.
    However, there are some tips for [reducing costs](#reducing-costs).

    ??? danger "Example Costs"

        For example, benchmarking one framework on the classification and regression suites
        on a one hour budget takes 1 hour * 10 folds * 100 datasets = 1,000 hours, plus
        overhead. Even when using spot instance pricing on `m5.2xlarge` instances (default)
        probably costs at least $100 US (prices depend on overhead and fluctating prices).
        A full evaluation with multiple frameworks and/or time budgets can cost
        thousands of dollars. 


## Setup

To run a benchmark on AWS you additionally need to have a configured AWS account.
The application is using the [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
Python package to exchange files through S3 and create EC2 instances.

If this is your first time setting up your AWS account on the machine that will run the 
`automlbenchmark` app, you can use the [AWS CLI](http://aws.amazon.com/cli/) tool and run:
 ```bash
 aws configure
 ```
You will need your AWS Access Key ID, AWS Secret Access Key, and pick a default [EC2 region](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-regions-availability-zones.html#concepts-available-regions).

!!! note "Selecting a Region" 
    To use a region, an AMI must be configured in the automl benchmark configuration file
    under `aws.ec2.regions`. The default configuration has AMIs for `us-east-1`, 
    `us-east-2`, `us-west-1`, `eu-west-1`, and `eu-central-1`. If you default EC2
    region is different from these, you will need to add the AMI to your [custom configuration](configuration.md#custom-configurations).
  
On first use, it is recommended to use the following configuration file, or to extend
your custom configuration file with these options. Follow the instructions in the file
and make any necessary adjustments before running the benchmark.

```yaml title="Starting AWS Configuration"
--8<-- "examples/aws/config.yaml"
```

To run a test to see if the benchmark framework is working on AWS, do the following:

```bash
python3 runbenchmark.py constantpredictor test -m aws
```

This will create and start an EC2 instance for each benchmark job and run the 6 jobs 
(3 OpenML tasks * 2 folds) from the `test` benchmark sequentially.
Each job will run is constrained to a one-minute limit in this case, excluding setup 
time for the EC2 instances (though `constantpredictor` will likely only take seconds).

For longer benchmarks, you'll probably want to run multiple jobs in parallel and 
distribute the work to several EC2 instances, for example:
```bash
python3 runbenchmark.py autosklearn validation 1h4c -m aws -p 4
```
will keep 4 EC2 instances running, monitor them in a dedicated thread, and finally collect all outputs from s3.

??? info "EC2 Instances always stopped eventually (by default)"

    Each EC2 instance is provided with a time limit at startup to ensure that in any case, 
    the instance is stopped even if there is an issue when running the benchmark task. 
    In this case the instance is stopped, not terminated, and we can therefore inspect 
    the machine manually (ideally after resetting its UserData field to avoid 
    re-triggering the benchmark on the next startup).

The console output is still showing the instances starting, outputs the progress and 
then the results for each dataset/fold combination (log excerpt from different command):

```{.text .limit_max_height title="Example output benchmarking H2O on AWS"}
Running `H2OAutoML_nightly` on `validation` benchmarks in `aws` mode!
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


## Configurable AWS Options

When using AWS mode, the application will use `on-demand` EC2 instances from the `m5` 
series by default. However, it is also possible to use `Spot` instances, specify a 
`max_hourly_price`, or customize your experience when using this mode in general.
All configuration points are grouped and documented under the `aws` yaml namespace in 
the main [config](GITHUB/resources/config.yaml) file.
When setting  your own configuration, it is strongly recommended to first create your 
own `config.yaml` file as described in [Custom configuration](configuration.md#custom-configurations).
Here is an example of a config file using Spot instances on a non-default region:
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

### Reducing Costs

The most important thing you can do to reduce costs is to critically evaluate which
experimental results can be re-used from previous publications. That said, when
conducting new experiments on AWS we have the following recommendations to reduce costs:

 - Use spot instances with a fixed maximum price: set `aws.ec2.spot.enabled: true` and `aws.ec2.spot.max_hourly_price`. 
   Check which region has [the lowest spot instance prices](https://aws.amazon.com/ec2/spot/)
   and configure `aws.region` accordingly. 
 - Skip the framework installation process by providing a docker image and setting `aws.docker_enabled: true`.
 - Set up [AWS Budgets](https://aws.amazon.com/aws-cost-management/aws-budgets/)
   to get alerts early if forecasted usage exceeds the budget. It should also be
   technically possibly to automatically shut down all running instances in a region
   if a budget is exceeded, but this naturally leads to a loss of experimental results, so
   it is best avoided.