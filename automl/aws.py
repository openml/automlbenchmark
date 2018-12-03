import logging
import os
import time

import boto3
import botocore.exceptions

from .benchmark import Benchmark, Job
from .docker import DockerBenchmark
from .resources import config as rconfig
from .utils import backup_file, datetime_iso, str_def, tail


log = logging.getLogger(__name__)


class AWSBenchmark(Benchmark):
    """AWSBenchmark
    an extension of Benchmark class, to run benchmarks on AWS
    """

    def __init__(self, framework_name, benchmark_name, parallel_jobs=1, region=None):
        """

        :param framework_name:
        :param benchmark_name:
        :param parallel_jobs:
        :param region:
        """
        super().__init__(framework_name, benchmark_name, parallel_jobs)
        self.region = region if region \
            else rconfig().aws.region if rconfig().aws['region'] \
            else boto3.session.Session().region_name
        self.ami = rconfig().aws.ec2.regions[self.region].ami
        self.s3 = None
        self.ec2 = None
        self.bucket = None
        self.instances = {}
        self.jobs = []
        self._validate2()

    def _validate(self):
        if self.parallel_jobs == 0 or self.parallel_jobs > rconfig().max_parallel_jobs:
            log.warning("forcing parallelization to its upper limit: %s", rconfig().max_parallel_jobs)
            self.parallel_jobs = rconfig().max_parallel_jobs

    def _validate2(self):
        if self.ami is None:
            raise ValueError("region {} not supported by AMI yet.".format(self.region))

    def setup(self, mode):
        if mode == Benchmark.SetupMode.skip:
            log.warning("AWS setup mode set to unsupported {mode}, ignoring".format(mode=mode))
        self.s3 = boto3.resource('s3', region_name=self.region)
        self.bucket = self._create_s3_bucket()
        self._upload_resources()
        self.ec2 = boto3.resource("ec2", region_name=self.region)

    def cleanup(self):
        self._stop_all_instances()
        self._delete_s3_bucket()

    def run(self, save_scores=False):
        # todo: parallelization improvement -> in many situations, creating a job for each fold may end up being much slower
        #   than having a job per task. This depends on job duration especially
        jobs = []
        if self.parallel_jobs == 1:
            jobs.append(self._make_job())
        else:
            jobs.extend(self._benchmark_jobs())
        self._run_jobs(jobs)

    def run_one(self, task_name: str, fold: int, save_scores=False):
        jobs = []
        if self.parallel_jobs == 1 and (fold is None or (isinstance(fold, list) and len(fold) > 1)):
            jobs.append(self._make_job(task_name, fold))
        else:
            jobs.extend(self._custom_task_jobs(task_name, fold))
        self._run_jobs(jobs)
        # board = Scoreboard.for_task(task_name, framework_name=self.framework_name)

    def _fold_job(self, task_def, fold: int):
        return self._make_job(task_def.name, [fold])

    def _make_job(self, task_name=None, folds=None):
        folds = [] if folds is None else [str(f) for f in folds]
        instance_type = self._get_task_def(task_name).ec2_instance_type if task_name else rconfig().aws.ec2.instance_type

        def _run():
            instance_id = self._start_instance(
                instance_type,
                "{framework} {benchmark} {task_param} {folds_param}".format(
                    framework=self.framework_name,
                    # benchmark=self.benchmark_name,
                    benchmark="/s3bucket/input/{}.json".format(self.benchmark_name),
                    task_param='' if task_name is None else ('-t '+task_name),
                    folds_param='' if len(folds) == 0 else ' '.join(['-f']+folds)
                ))
            self._monitor_instance(instance_id)

        job = Job("aws_{}_{}_{}".format(task_name, ':'.join(folds), self.framework_name))
        job._run = _run
        return job

    def _start_instance(self, instance_type, script_params):
        log.info("Starting new EC2 instance with params: %s", script_params)
        inst_key = "{}_${}_i{}".format(self.ami, self.uid, datetime_iso(micros=True, no_sep=True)).lower()
        instance = self.ec2.create_instances(
            ImageId=self.ami,
            MinCount=1,
            MaxCount=1,
            InstanceType=instance_type,
            SubnetId=rconfig().aws.ec2.subnet_id,
            UserData=self._ec2_startup_script(inst_key, script_params)
        )[0]
        # todo: error handling
        self.instances[instance.id] = (instance, inst_key)
        log.info("Started EC2 instance %s", instance.id)
        return instance.id

    def _monitor_instance(self, instance_id):
        instance, _ = self.instances[instance_id]

        def log_console():
            try:
                output = instance.console_output(Latest=True)
                if 'Output' in output:
                    output = output['Output']
                    log.info(tail(output, 50))
            except Exception as e:
                log.exception(e)

        while True:
            log.info("[%s] checking %s: %s", datetime_iso(), instance_id, instance.state['Name'])
            if instance.state['Code'] > 16:     # ended instance
                log.info("EC2 instance %s is %s", instance_id, instance.state['Name'])
                log_console()
                self._download_results(instance_id)
                del self.instances[instance_id]
                break
            else:
                log_console()
            time.sleep(rconfig().aws.query_frequency_seconds)

    def _stop_instance(self, instance_id, terminate=False):
        log.info("%s EC2 instances %s", "Terminating" if terminate else "Stopping", instance_id)
        instance, _ = self.instances[instance_id]
        if terminate:
            response = instance.terminate()
        else:
            response = instance.stop()
        log.info("%s EC2 instances %s with response %s", "Terminated" if terminate else "Stopped", instance_id, response)
        # todo error handling

    def _stop_all_instances(self, terminate=False):
        ids = [inst.id for inst in self.instances]
        log.info("%s EC2 instances %s", "Terminating" if terminate else "Stopping", ids)
        instances = self.ec2.instances.filter(InstanceIds=ids)
        if terminate:
            response = instances.terminate()
        else:
            response = instances.stop()
        log.info("%s EC2 instances %s with response %s", "Terminated" if terminate else "Stopped", ids, response)
        # todo error handling

    def _create_s3_bucket(self):
        bucket_name = rconfig().aws.s3.bucket
        if rconfig().aws.s3.temporary:
            bucket_name += ('-' + self.uid)
        try:
            self.s3.meta.client.head_bucket(Bucket=bucket_name)
            bucket = self.s3.Bucket(bucket_name)
        except botocore.exceptions.ClientError as e:
            error_code = int(e.response['Error']['Code'])
            if error_code == 404:
                log.info("%s bucket doesn't exist in region %s, creating it", bucket_name, self.region)
                bucket = self.s3.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration=dict(
                        LocationConstraint=self.region
                    )
                )
        return bucket

    def _delete_s3_bucket(self):
        if self.bucket and rconfig().aws.s3.temporary:
            self.bucket.delete()

    def _upload_resources(self):
        root_key = str_def(rconfig().aws.s3.root_key)
        benchmark_basename = os.path.basename(self.benchmark_path)
        self.bucket.upload_file(self.benchmark_path, root_key+('/'.join(['input', benchmark_basename])))

    def _download_results(self, instance_id):
        instance, ikey = self.instances[instance_id]
        root_key = str_def(rconfig().aws.s3.root_key)
        predictions_objs = [o for o in self.bucket.objects.filter(Prefix=root_key+('/'.join(['output', ikey, 'predictions'])))]
        scores_objs = [o for o in self.bucket.objects.filter(Prefix=root_key+('/'.join(['output', ikey, 'scores'])))]

        for obj in predictions_objs:
            # it should be safe and good enough to simply save predictions file as usual (after backing up previous prediction)
            dest_path = os.path.join(rconfig().predictions_dir, os.path.basename(obj.key))
            backup_file(dest_path)
            obj.download_file(dest_path)
        for obj in scores_objs:
            # todo: saving scores for now after backing up previous scores but this should be merged to existing scores files!!!
            dest_path = os.path.join(rconfig().scores_dir, os.path.basename(obj.key))
            backup_file(dest_path)
            obj.download_file(dest_path)

    def _ec2_startup_script(self, instance_key, script_params):
        # todo: quid of docker repo access? public image?
        # note for power_state: delay (in mn) passed to the shutdown command is executed, timeout (in sec) waiting for cloud-init to complete before triggering shutdown
        return """
#cloud-config

package_update: true
package_upgrade: false
packages:
  - curl
  - wget
  - unzip
  - awscli
  - git
  - python3
  - python3-pip
  - docker

runcmd:
  - alias PIP='pip3'
  - alias PY='python3 -W ignore'
  - mkdir -p /s3bucket/input
  - mkdir -p /s3bucket/output
  - mkdir /repo
  - cd /repo
  - git clone {repo} .
  - PIP install --no-cache-dir -r requirements.txt --process-dependency-links
  - PIP install --no-cache-dir openml
  - aws s3 cp {s3_base_url}/input /s3bucket/input --recursive
#  - PY {script} {params} -i /s3bucket/input -o /s3bucket/output -s only 
  - PY {script} {params} -i /s3bucket/input -o /s3bucket/output -m docker -s skip
  - aws s3 cp /s3bucket/output {s3_base_url}/output/{ikey} --recursive
  - rm -f /var/lib/cloud/instances/*/sem/config_scripts_user

final_message: "AutoML benchmark completed after $UPTIME s"

power_state:
  delay: "+1"
  mode: poweroff
  message: See you soon
  timeout: 21600
  condition: True
 
""".format(
            repo=rconfig().project_repository,
            s3_base_url="s3://{bucket}/{root}".format(bucket=self.bucket.name, root=str_def(rconfig().aws.s3.root_key)),
            script=rconfig().script,
            ikey=instance_key,
            params=script_params
        )

    def _ec2_startup_script_bash(self, instance_key, script_params):
        return """#!/bin/bash
apt-get update
#apt-get -y upgrade
apt-get install -y curl wget unzip git awscli
apt-get install -y python3 python3-pip 
pip3 install --upgrade pip

alias PIP='pip3'
alias PY='python3 -W ignore'

mkdir -p /s3bucket/input
mkdir -p /s3bucket/output
mkdir ~/repo
cd ~/repo
git clone {repo} .

PIP install --no-cache-dir -r requirements.txt --process-dependency-links
PIP install --no-cache-dir openml

aws s3 cp {s3_base_url}/input /s3bucket/input --recursive
#PY {script} {params} -o /s3bucket -s only
PY {script} {params} -o /s3bucket -s skip
aws s3 cp /s3bucket/output {s3_base_url}/output/{ikey} --recursive
rm -f /var/lib/cloud/instances/*/sem/config_scripts_user
""".format(
            repo=rconfig().project_repository,
            s3_base_url="s3://{bucket}/{root}".format(bucket=self.bucket.name, root=str_def(rconfig().aws.s3.root_key)),
            script=rconfig().script,
            ikey=instance_key,
            params=script_params
        )


class AWSRemoteBenchmark(DockerBenchmark):

    # TODO: idea is to handle results progressively on the remote side and push results as soon as they're generated
    #   this would allow to safely run multiple tasks on single AWS instance

    def __init__(self, framework_name, benchmark_name, parallel_jobs=1, region=None):
        self.region = region
        self.s3 = boto3.resource('s3', region_name=self.region)
        self.bucket = self._init_bucket()
        self._download_resources()
        super().__init__(framework_name, benchmark_name, parallel_jobs)

    def run(self, save_scores=False):
        super().run(save_scores)
        self._upload_results()

    def run_one(self, task_name: str, fold, save_scores=False):
        super().run_one(task_name, fold, save_scores)
        self._upload_results()

    def _make_job(self, task_name=None, folds=None):
        job = super()._make_job(task_name, folds)
        super_run = job._run
        def new_run():
            super_run()
            # self._upload_result()

        job._run = new_run
        return job

    def _init_bucket(self):
        pass

    def _download_resources(self):
        root_key = str_def(rconfig().aws.s3.root_key)
        benchmark_basename = os.path.basename(self.benchmark_path)
        self.bucket.upload_file(self.benchmark_path, root_key+('/'.join(['input', benchmark_basename])))

    def _upload_results(self):
        pass

