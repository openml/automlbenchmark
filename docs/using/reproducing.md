# Reproducing a Benchmark Evaluation

This guide distinguishes three levels of reproducibility, which we will informally call *loose*, *balanced*, and *strict*.

## Loose Reproducibility
A loose reproduction means evaluating a specific framework on a specific task, using the modern version of the AutoML benchmark.
It is often possible to evaluate the desired framework with the most recent version of the benchmark.
It might even be possible to evaluate the specific (older) version of the AutoML framework of the work you want to reproduce.
Unless the framework received significant updates, this will generally result in a very similar result as the original.
The advantage of this is that you can just use the benchmark in the way you are used to (as described in "Getting Started"),
as most work on the benchmark itself does not impact the evaluation results, but you can use up-to-date features and documentation.
To improve the faithfulness of the reproducibility, use docker mode to constrain the frameworks resources more rigorously.

## Balanced Reproducibility
A balanced reproduction aims to provide a very faithful reproduction while avoiding a lot of the small details which require a lot of work but are almost certainly not going to lead to different results.
For one, you use the most recent release with the same _minor_ version. For the JMLR paper, that includes experiments on versions *2.1.0* through *2.1.7*, this means only using *2.1.7* for all experiments.
Generally speaking, patch-level releases only address bugfixes or contain changes which do not affect the outcome of the experiments.

Similarly, we do not set any random seeds. In the AutoML Benchmark, random seeds are used just by the AutoML frameworks themselves.
The 10-fold cross-validation splits themselves are not determined by a random seed, but are instead consistent and provided by OpenML (or, alternatively, are defined in a file).
By default, the AutoML benchmark will provide a different random seed to the framework for each fold in the evaluation.
This means that the effect of any one random seed will not be large. As such, we expect to find similar results regardless of whether seeds are set or not.

## Strict Reproducibility
Strict reproducibility means a true best effort to recreate the exact same setup that the original had.
However, even with that accounted for, keep in mind that you will not likely get the exact same results because of factors like:

 - Using slightly different hardware (even AWS EC2 instances are heterogeneous).
 - Uncontrolled randomness in AutoML frameworks, such as those that arise from race conditions in multiprocess applications.
 - Hardware errors and limitations, such as numerical precision errors.

That said, to reproduce the setup as closely as possible, you need access to the configurations that are used and the results file.
The results file specifies for each job which AMLB version, framework version, random seed, and so on, were used.
Using the same configuration file and the same AMLB version installed, you can then run a command like:

```commandline
python runbenchmark.py FRAMEWORK:VERSION CONSTRAINT STUDY --task=TASK_NAME --fold=FOLD_NO --seed=SEED -m aws
```

??? note "In the future..." 

    We do want to improve support for rerunning experiments defined in a `results.csv` file format. 
    This is useful not just for reproducibility, but also for running failed jobs, or scheduling a job matrix.
    However, we currently do not have the resources to add this feature. We welcome contributions.

# Reproducing the JMLR Paper
When opting for a *balanced* evaluation, it should be sufficient to use benchmark version *2.1.7*.
Version *2.1.0* denotes the last minor benchmark update, and all subsequent releases are bugfixes to address issues that came up during evaluation.
Since most bugs only affected specific tasks or frameworks, only those jobs which were affected by the following patch-releases were run in later versions.
Install version *2.1.7* and evaluate each framework on a 32GiB, 8 vCPU machine with the command:

```commandline
python runbenchmark.py FRAMEWORK:2023Q2 CONSTRAINT STUDY -m docker
```
Note that `FRAMEWORK` here is any of the predefined configurations of `resources/frameworks_2023Q2.yaml`,
`CONSTRAINT`s used were `1h8c_gp3` and `4h8c_gp3`, and `STUDY` was one of `openml/s/269` or `/openml/s/271`.

We used AWS's `m5.2xlarge` instances for this (using `-m aws` instead of `-m docker`), so we could also parallelize the evaluations (`-p` parameter).
If you do too, make sure to update your configuration accordingly, so you use the right AMLB version and instance type.
See the AWS configuration section for how to get started with the AMLB on AWS, and be warned that this can get expensive quickly.

??? abstract "Example of a custom configuration files"

    ```yaml title="~/.config/automlbenchmark/config.yaml"
    # put this file in your ~/.config/automlbenchmark directory
    # to override default configs
    ---
    project_repository: https://github.com/openml/automlbenchmark#v2.1.7
    benchmarks:                     # configuration namespace for the benchmarks definitions.
      definition_dir:               # list of directories containing the benchmarks yaml definitions.
        - '{root}/resources/benchmarks'
        - '{user}/benchmarks'
      constraints_file:
        - '{user}/constraints.yaml'
        - '{root}/resources/constraints.yaml'

    versions:
      python: 3.9

    aws:
      use_docker: true
      iam:
        temporary: false  # set to true if you want IAM entities (credentials used by ec2 instances) being recreated for each benchmark run.
        credentials_propagation_waiting_time_secs: 360  # increase this waiting time if you encounter credentials issues on ec2 instances when using temporary IAM.

      s3:
        bucket: NAME-OF-BUCKET-WE-USED # automl-benchmark-bucket # ALWAYS SET this bucket name as it needs to be unique in entire S3 domain. #  automl-benchmark-697442f1
                                            # (40 chars max as the app reserves some chars for temporary buckets)
                                            # if you prefer using temporary s3 buckets (see below), you can comment out this property.
        temporary: false  # set to true if you want a new s3-bucket being temporarily created/deleted for each benchmark run.

      ec2:
        terminate_instances: always  # see resources/config.yaml for explanations: you may want to switch this value to `success` if you want to investigate on benchmark failures.
        spot:
          enabled: true
          max_hourly_price: 0.20  # comment out to use defaulti
        monitoring:
          cpu:
            query_interval_seconds: 900
            abort_inactive_instances: false
        regions:
          eu-north-1:
            ami: ami-0989fb15ce71ba39e

      resource_files:  # this allows to automatically upload custom config + frameworks to the running instance (benchmark files are always uploaded).
        - '{user}/config.yaml'
        - '{user}/frameworks.yaml'
        - '{user}/extensions'
        - '{user}/benchmarks'
        - '{user}/constraints.yaml'

      job_scheduler:
        retry_on_errors:                         # Boto3 errors that will trigger a job reschedule.
          - 'SpotMaxPriceTooLow'
          - 'MaxSpotInstanceCountExceeded'
          - 'InsufficientFreeAddressesInSubnet'
          - 'InsufficientInstanceCapacity'
          - 'RequestLimitExceeded'
          - 'VolumeLimitExceeded'
          - 'VcpuLimitExceeded'
        retry_on_states:                         # EC2 instance states that will trigger a job reschedule.
          - 'Server.SpotInstanceShutdown'
          - 'Server.SpotInstanceTermination'
          - 'Server.InsufficientInstanceCapacity'
          - 'Client.VolumeLimitExceeded'
          - 'VcpuLimitExceeded'


    job_scheduler:
      max_parallel_jobs: 1000
      delay_between_jobs: 8

    inference_time_measurements:
      enabled: false
      additional_job_time: 1800

    frameworks:              # configuration namespace for the frameworks definitions.
      definition_file:       # list of yaml files describing the frameworks base definitions.
        - '{root}/resources/frameworks.yaml'
        - '{user}/frameworks.yaml'
    ```




# Installing an Old Version
For the most part, installing old versions is identical to installing the latest version.
There is one exception to this rule, which is the supported Python version.
The AMLB 2.0 versions work with Python 3.6, the AMLB 2.1 releases work with Python 3.9.
Otherwise, simply remember to check out the right release before installing and if you encounter any discrepancies please refer to the documentation of that version.