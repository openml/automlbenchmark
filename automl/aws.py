import logging

import boto3

from .benchmark import Benchmark


log = logging.getLogger(__name__)


class AWSBenchmark(Benchmark):
    """AWSBenchmark
    an extension of Benchmark class, to run benchmarks on AWS
    """

    def __init__(self, framework_name, benchmark_name, resources, region=None, reuse_instance=False):
        """

        :param framework_name:
        :param benchmark_name:
        :param resources:
        :param region:
        :param reuse_instance:
        """
        super().__init__(framework_name, benchmark_name, resources)
        self.region = region if region \
            else self.resources.config.aws.default_region if self.resources.config.aws['default_region'] \
            else boto3.session.Session().region_name
        self.ec2 = boto3.resource("ec2", region_name=self.region)
        self.instances = []
        self.reuse_instance = reuse_instance

        self.ami = self.resources.config.aws.regions[self.region].ami
        if self.ami is None:
            raise ValueError("Region not supported by AMI yet.")

    def setup(self, mode):
        if mode == Benchmark.SetupMode.skip:
            return

    def run(self):
        if self.reuse_instance:
            self.start_instances("{framework} {benchmark}".format(
                script=self.resources.config['script'],
                framework=self.framework_def.name,
                benchmark=self.benchmark_name
            ))
        else:
            super().run()

    def _run_fold(self, task_def, fold: int):
        self.run_one(task_def.name, fold)

    def run_one(self, task_name: str, fold: int):
        self.start_instances("{framework} {benchmark} -t {task} -f {fold}".format(
            framework=self.framework_def.name,
            benchmark=self.benchmark_name,
            task=task_name,
            fold=fold
        ))

    def start_instances(self, script_params):
        log.info("Starting new EC2 instances with params: %s", script_params)
        instances = self.ec2.create_instances(
            ImageId=self.ami,
            MinCount=1,
            MaxCount=1,
            InstanceType=self.benchmark_def.aws_instance_type,
            SubnetId=self.resources.config.aws.subnet_id,
            UserData=self._startup_script(script_params)
        )[0]
        self.instances.extend(instances)
        log.info("Started EC2 instances %s", [inst.id for inst in instances])

    def stop_instances(self, ids=None):
        ids = ids if ids else [inst.id for inst in self.instances]
        # self.ec2.instances.filter(InstanceIds=ids).stop()
        log.info("Terminating EC2 instances %s", ids)
        response = self.ec2.instances.filter(InstanceIds=ids).terminate()
        log.info("Terminated EC2 instances with response %s", response)
        # todo error handling

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
            repo=self.resources.config.project_repository,
            script=self.resources.config.script,
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
            repo=self.resources.config.project_repository,
            script=self.resources.config.script,
            params=params
        )


