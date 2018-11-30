import logging
import time

import boto3

from .benchmark import Benchmark, Job
from .resources import config as rconfig
from .utils import now_iso


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

        def _run():
            self._start_instance("{framework} {benchmark} {task_param} {folds_param}".format(
                framework=self.framework_name,
                benchmark=self.benchmark_name,  # todo: pass path to downloaded benchmark def file
                task_param='' if task_name is None else ('-t '+task_name),
                folds_param='' if len(folds) == 0 else ' '.join(['-f']+folds)
            ))

        job = Job("aws_{}_{}_{}".format(task_name, ':'.join(folds), self.framework_name))
        job._run = _run
        return job

    def _run_jobs(self, jobs):
        while len(self.instances) > 0:
            for iid, (instance, ikey) in self.instances.items():
                if instance.state['Code'] > 16:     # ended instance
                    log.info("EC2 instance %s is %s", iid, instance.state['Name'])
                    self._download_results(iid)
                    del self.instances[iid]
            time.sleep(rconfig().aws.query_frequency_seconds)

    def _start_instance(self, script_params):
        log.info("Starting new EC2 instance with params: %s", script_params)
        inst_key = "{}_${}_i{}".format(self.ami, self.uid, now_iso(micros=True, no_sep=True))
        instance = self.ec2.create_instance(
            ImageId=self.ami,
            MinCount=1,
            MaxCount=1,
            InstanceType=self.benchmark_def.ec2_instance_type,
            SubnetId=rconfig().aws.ec2.subnet_id,
            UserData=self._ec2_startup_script(inst_key, script_params)
        )[0]
        # todo: error handling
        self.instances[instance.id] = (instance, inst_key)
        log.info("Started EC2 instance %s", instance.id)

    def _monitor_instance(self, instance):
        # todo: ideally, would be nice to monitor instance individually and asynchronously, cf. asyncio
        pass

    def _stop_instance(self, instance, terminate=False):
        log.info("%s EC2 instances %s", "Terminating" if terminate else "Stopping", instance.id)
        if terminate:
            response = instance.terminate()
        else:
            response = instance.stop()
        log.info("%s EC2 instances %s with response %s", "Terminated" if terminate else "Stopped", instance.id, response)
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

    def _upload_resources(self):
        # todo: upload benchmark definition to bucket
        pass

    def _download_results(self, instance_id):
        # todo: ensure we download all the files and only the files created by this run
        #   idea: if s3 if shared, then pass self.uid to instances and create upload dir named after self.uid on s3
        pass

    @staticmethod
    def _ec2_startup_script(instance_key, script_params):
        # todo:
        #   packages:
        #     ensure docker is installed
        #     quid of docker repo access? public image?
        #   in runcmd:
        #     export S3_PATH=s3://{self.bucket.name}/
        #     download resources from S3_PATH to /s3bucket/input/resources
        #     pass /s3bucket/input as input dir to docker: -i /s3bucket/input
        #     pass /s3bucket/output as output dir to docker: -o /s3bucket/output
        #     pass full path to benchmark file to docker: /s3bucket/input/resources/{benchmark}.json
        #     start docker image (skip mode if we always want a prebuilt image)
        #     on completion:
        #       upload /s3bucket/output/predictions to s3
        #       upload /s3bucket/output/scores to s3
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
  - export PIP=pip3
  - export -f PY() {{ python3 -W ignore "$@" \}}
  - mkdir -p /s3bucket
  - mkdir ~/repo
  - cd ~/repo
  - git clone {repo} .
  - PIP install --no-cache-dir -r requirements.txt --process-dependency-links
  - PIP install --no-cache-dir openml
  - PY {script} {params} -i /s3bucket/input -o /s3bucket/output -s only
  - PY {script} {params} -i /s3bucket/input -o /s3bucket/output -s skip
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
            params=script_params
        )

    @staticmethod
    def _ec2_startup_script_bash(instance_key, script_params):
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
            params=script_params
        )


