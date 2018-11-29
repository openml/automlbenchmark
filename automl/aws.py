import logging

import boto3

from .benchmark import Benchmark
from .resources import config as rconfig


log = logging.getLogger(__name__)


class AWSBenchmark(Benchmark):
    """AWSBenchmark
    an extension of Benchmark class, to run benchmarks on AWS
    """

    def __init__(self, framework_name, benchmark_name, region=None, keep_instance=False):
        """

        :param framework_name:
        :param benchmark_name:
        :param region:
        :param keep_instance:
        """
        super().__init__(framework_name, benchmark_name)
        self.region = region if region \
            else rconfig().aws.region if rconfig().aws['region'] \
            else boto3.session.Session().region_name
        self.ami = rconfig().aws.ec2.regions[self.region].ami
        if self.ami is None:
            raise ValueError("Region not supported by AMI yet.")
        self.s3 = None
        self.ec2 = None
        self.bucket = None
        self.instances = []
        self.keep_instance = keep_instance

    def setup(self, mode):
        if mode == Benchmark.SetupMode.skip:
            log.warning("AWS setup mode set to unsupported {mode}, ignoring".format(mode=mode))
        self.s3 = boto3.resource('s3', region_name=self.region)
        self.bucket = self._create_s3_bucket()
        self.ec2 = boto3.resource("ec2", region_name=self.region)

    def cleanup(self):
        self._stop_instances()
        self._delete_s3_bucket()

    def run(self, save_scores=False):
        if self.keep_instance:
            self._start_instances("{framework} {benchmark}".format(
                script=rconfig()['script'],
                framework=self.framework_def.name,
                benchmark=self.benchmark_name
            ))
        else:
            super().run(save_scores=False)

    def _run_fold(self, task_def, fold: int):
        self.run_one(task_def.name, fold)

    def run_one(self, task_name: str, fold: int, save_scores=False):
        self._start_instances("{framework} {benchmark} -t {task} -f {fold}".format(
            framework=self.framework_def.name,
            benchmark=self.benchmark_name,
            task=task_name,
            fold=fold
        ))

    def _start_instances(self, script_params):
        log.info("Starting new EC2 instances with params: %s", script_params)
        instances = self.ec2.create_instances(
            ImageId=self.ami,
            MinCount=1,
            MaxCount=1,
            InstanceType=self.benchmark_def.ec2_instance_type,
            SubnetId=rconfig().aws.ec2.subnet_id,
            UserData=self._startup_script(script_params)
        )[0]
        self.instances.extend(instances)
        log.info("Started EC2 instances %s", [inst.id for inst in instances])

    def _stop_instances(self, ids=None):
        ids = ids if ids else [inst.id for inst in self.instances]
        # self.ec2.instances.filter(InstanceIds=ids).stop()
        log.info("Terminating EC2 instances %s", ids)
        response = self.ec2.instances.filter(InstanceIds=ids).terminate()
        log.info("Terminated EC2 instances with response %s", response)
        # todo error handling

    def _create_s3_bucket(self):
        bucket_name = rconfig().aws.s3.bucket
        if rconfig().aws.s3.temporary:
            bucket_name += ('_' + self.uid)
        if bucket_name in self.s3.get_available_subresources():
            return self.s3.Bucket(bucket_name)  # apparently no need to load bucket
        else:
            return self.s3.create_bucket(
                Bucket=bucket_name
            )

    def _delete_s3_bucket(self):
        if self.bucket and rconfig().aws.s3.temporary:
            self.bucket.delete()

    def _download_results(self):
        # todo: ensure we download all the files and only the files created by this run
        # idea: if s3 if shared, then pass self.uid to instances and create upload dir named after self.uid on s3
        pass

    def _startup_script(self, params):
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

runcmd:
  - export PIP=pip3
  - export -f PY() {{ python3 -W ignore "$@" \}}
  - mkdir -p /s3bucket
  - mkdir ~/repo
  - cd ~/repo
  - git clone {repo} .
  - PIP install --no-cache-dir -r requirements.txt --process-dependency-links
  - PIP install --no-cache-dir openml
  - PY {script} {params} -o /s3bucket -s only
  - PY {script} {params} -o /s3bucket -s skip
  - rm -f /var/lib/cloud/instances/*/sem/config_scripts_user

final_message: "AutoML benchmark completed after $UPTIME s"

power_state:
  delay: "+30"
  mode: poweroff
  message: See you soon
  timeout: 21600
  condition: True
 
""".format(
            repo=rconfig().project_repository,
            script=rconfig().script,
            params=params
        )

    def _startup_script_bash(self, params):
        return """#!/bin/bash
apt-get update
#apt-get -y upgrade
apt-get install -y curl wget unzip git awscli
apt-get install -y python3 python3-pip 
pip3 install --upgrade pip

export PIP=pip3
export -f PY() {{ python3 -W ignore "$@" \}}

mkdir /s3bucket
mkdir ~/repo
cd ~/repo
git clone {repo} .

PIP install --no-cache-dir -r requirements.txt --process-dependency-links
PIP install --no-cache-dir openml

PY {script} {params} -o /s3bucket -s only
PY {script} {params} -o /s3bucket -s skip
rm -f /var/lib/cloud/instances/*/sem/config_scripts_user
""".format(
            repo=rconfig().project_repository,
            script=rconfig().script,
            params=params
        )


