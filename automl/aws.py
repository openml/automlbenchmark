"""
**aws** module is built on top of **benchmark** to provide the platform-specific logic
necessary to run a benchmark on EC2 instances:

- create a S3 bucket (if it doesn't already exist).
- upload some resources on S3.
- configures an AWS IAM profile to provide read/write access to the S3 bucket from the future EC2 instances.
- create jobs and start an EC2 instance for each job:
    - the EC2 instance download some resources from S3.
    - the EC2 instance runs the task locally or using docker.
    - on task completion, the EC2 instance uploads the results and logs to S3 and stops.
- monitors each job and downloads results and logs from s3 when the job is completed.
- merge downloaded results with existing/local results.
- properly cleans up AWS resources (S3, EC2).
"""
import io
import json
import logging
import math
import os
import re
import time

import boto3
import botocore.exceptions

from .benchmark import Benchmark
from .docker import DockerBenchmark
from .job import Job
from .resources import config as rconfig
from .results import Scoreboard
from .utils import backup_file, datetime_iso, list_all_files, str_def, tail


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
        self.suid = datetime_iso(micros=True, no_sep=True)  # short uid for AWS entities whose name length is limited
        self.region = region if region \
            else rconfig().aws.region if rconfig().aws['region'] \
            else boto3.session.Session().region_name
        self.ami = rconfig().aws.ec2.regions[self.region].ami
        self.ec2 = None
        self.iam = None
        self.s3 = None
        self.bucket = None
        self.uploaded_resources = None
        self.instance_profile = None
        self.instances = {}
        self.jobs = []
        self._validate2()

    def _validate(self):
        if rconfig().aws.ec2.terminate_instances not in ['always', 'success', 'never', True, False]:
            raise ValueError("`terminate_instances` setting should be one among ['always', 'success', 'never']")

        if self.parallel_jobs == 0 or self.parallel_jobs > rconfig().max_parallel_jobs:
            log.warning("Forcing parallelization to its upper limit: %s.", rconfig().max_parallel_jobs)
            self.parallel_jobs = rconfig().max_parallel_jobs

    def _validate2(self):
        if self.ami is None:
            raise ValueError("Region {} not supported by AMI yet.".format(self.region))

    def setup(self, mode):
        if mode == Benchmark.SetupMode.skip:
            log.warning("AWS setup mode set to unsupported {mode}, ignoring.".format(mode=mode))
        # S3 setup to exchange files between local and ec2 instances
        self.s3 = boto3.resource('s3', region_name=self.region)
        self.bucket = self._create_s3_bucket()
        self.uploaded_resources = self._upload_resources()

        # IAM setup to secure exchanges between s3 and ec2 instances
        self.iam = boto3.resource('iam', region_name=self.region)
        if mode == Benchmark.SetupMode.force:
            log.warning("Cleaning up previously created IAM entities if any.")
            self._delete_iam_entities()
        self.instance_profile = self._create_instance_profile()

        # EC2 setup to prepare creation of ec2 instances
        self.ec2 = boto3.resource("ec2", region_name=self.region)

    def cleanup(self):
        self._stop_all_instances()
        self._delete_resources()
        if rconfig().aws.iam.temporary is True:
            self._delete_iam_entities()
        if rconfig().aws.s3.temporary is True:
            self._delete_s3_bucket()

    def run(self, save_scores=False):
        # TODO: parallelization improvement -> in many situations, creating a job for each fold may end up being much slower
        #   than having a job per task. This depends on job duration especially
        jobs = []
        if self.parallel_jobs == 1:
            jobs.append(self._make_job())
        else:
            jobs.extend(self._benchmark_jobs())
        try:
            results = self._run_jobs(jobs)
        finally:
            self.cleanup()
        return self._process_results(results, save_scores=save_scores)

    def run_one(self, task_name: str, fold: int, save_scores=False):
        jobs = []
        if self.parallel_jobs == 1 and (fold is None or (isinstance(fold, list) and len(fold) > 1)):
            jobs.append(self._make_job(task_name, fold))
        else:
            task_def = self._get_task_def(task_name)
            jobs.extend(self._custom_task_jobs(task_def, fold))
        try:
            results = self._run_jobs(jobs)
        finally:
            self.cleanup()
        return self._process_results(results, task_name=task_name, save_scores=save_scores)

    def _fold_job(self, task_def, fold: int):
        return self._make_job(task_def.name, [fold])

    def _make_job(self, task_name=None, folds=None):
        folds = [] if folds is None else [str(f) for f in folds]
        instance_type = self._get_task_def(task_name).ec2_instance_type if task_name else rconfig().aws.ec2.instance_type
        timeout_secs = self._get_task_def(task_name).max_runtime_seconds if task_name \
            else sum([task.max_runtime_seconds for task in self.benchmark_def])
        timeout_secs += rconfig().aws.overhead_time_seconds

        job = Job('_'.join(['aws', self.benchmark_name, task_name if task_name else 'all', '.'.join(folds), self.framework_name]))
        job.instance_id = None

        def _run(job_self):
            resources_root = "" if rconfig().aws.use_docker else "/s3bucket"
            job_self.instance_id = self._start_instance(
                instance_type,
                script_params="{framework} {benchmark} {task_param} {folds_param}".format(
                    framework=self.framework_name,
                    benchmark="{}/user/{}.yaml".format(resources_root, self.benchmark_name),
                    task_param='' if task_name is None else ('-t '+task_name),
                    folds_param='' if len(folds) == 0 else ' '.join(['-f']+folds)
                ),
                instance_key='_'.join([job.name, datetime_iso(micros=True, time_sep='.')]),
                timeout_secs=timeout_secs
            )
            return self._wait_for_results(job_self)

        def _on_done(job_self):
            terminate = self._download_results(job_self.instance_id)
            if not terminate and rconfig().aws.ec2.terminate_instances == 'success':
                log.warning("[WARNING]: EC2 Instance %s won't be terminated as we couldn't download the results: "
                            "please terminate it manually or restart it (after clearing its UserData) if you want to inspect the instance.",
                            job_self.instance_id)
            self._stop_instance(job_self.instance_id, terminate=terminate)

        job._run = _run.__get__(job)
        job._on_done = _on_done.__get__(job)
        return job

    def _wait_for_results(self, job):
        instance, _ = self.instances[job.instance_id]
        last_console_line = -1

        def log_console():
            nonlocal last_console_line
            try:
                output = instance.console_output(Latest=True)
                if 'Output' in output:
                    output = output['Output']   # note that console_output only returns the last 64kB of console
                    new_log, last_line = tail(output, from_line=last_console_line, include_line=False)
                    if last_line is not None:
                        last_console_line = last_line['line']
                    if new_log:
                        log.info(new_log)
            except Exception as e:
                log.exception(e)

        while True:
            log.info("[%s] checking job %s on instance %s: %s.", datetime_iso(), job.name, job.instance_id, instance.state['Name'])
            log_console()
            if instance.state['Code'] > 16:     # ended instance
                log.info("EC2 instance %s is %s.", job.instance_id, instance.state['Name'])
                break
            time.sleep(rconfig().aws.query_frequency_seconds)

    def _start_instance(self, instance_type, script_params="", instance_key=None, timeout_secs=-1):
        log.info("Starting new EC2 instance with params: %s.", script_params)
        inst_key = instance_key.lower() if instance_key \
            else "{}_p{}_i{}".format(self.uid,
                                     re.sub(r"[\s-]", '', script_params),
                                     datetime_iso(micros=True, time_sep='.')).lower()
        # TODO: don't know if it would be considerably faster to reuse previously stopped instances sometimes
        #   instead of always creating a new one:
        #   would still need to set a new UserData though before restarting the instance.
        instance = self.ec2.create_instances(
            ImageId=self.ami,
            MinCount=1,
            MaxCount=1,
            InstanceType=instance_type,
            SubnetId=rconfig().aws.ec2.subnet_id,
            IamInstanceProfile=dict(Name=self.instance_profile.name),
            UserData=self._ec2_startup_script(inst_key, script_params=script_params, timeout_secs=timeout_secs)
        )[0]
        # TODO: error handling
        self.instances[instance.id] = (instance, inst_key)
        log.info("Started EC2 instance %s.", instance.id)
        return instance.id

    def _stop_instance(self, instance_id, terminate=None):
        terminate_config = rconfig().aws.ec2.terminate_instances
        if terminate_config in ['always', True]:
            terminate = True
        elif terminate_config in ['never', False]:
            terminate = False
        else:
            terminate = False if terminate is None else terminate

        log.info("%s EC2 instances %s.", "Terminating" if terminate else "Stopping", instance_id)
        instance, _ = self.instances[instance_id]
        del self.instances[instance_id]
        try:
            if terminate:
                response = instance.terminate()
            else:
                response = instance.stop()
            log.info("%s EC2 instances %s with response %s.", "Terminated" if terminate else "Stopped", instance_id, response)
        except Exception as e:
            log.error("ERROR: EC2 instance %s could not be %s!\n"
                      "Even if the instance should stop by itself after a certain timeout, "
                      "you may want to stop/terminate it manually:\n%s",
                      instance_id, "terminated" if terminate else "stopped", str(e))
        finally:
            try:
                log.info("Instance %s state: %s.", instance_id, response['TerminatingInstances'][0]['CurrentState']['Name'])
            except:
                pass

    def _stop_all_instances(self):
        for id in list(self.instances.keys()):
            self._stop_instance(id)

    def _create_s3_bucket(self):
        # cf. s3 restrictions: https://docs.aws.amazon.com/AmazonS3/latest/dev/BucketRestrictions.html
        bucket_name = rconfig().aws.s3.bucket
        if rconfig().aws.s3.temporary:
            bucket_name += ('-' + self.suid)
        try:
            self.s3.meta.client.head_bucket(Bucket=bucket_name)
            bucket = self.s3.Bucket(bucket_name)
        except botocore.exceptions.ClientError as e:
            error_code = int(e.response['Error']['Code'])
            if error_code == 404:
                log.info("%s bucket doesn't exist, creating it in region %s.", bucket_name, self.region)
                bucket = self.s3.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration=dict(
                        LocationConstraint=self.region
                    )
                )
                log.info("S3 bucket %s was successfully created.", bucket_name)
        return bucket

    def _delete_s3_bucket(self):
        if self.bucket:
            # we can only delete 1000 objects at a time using this API,
            # but this is intended only for temporary buckets, so no need for pagination
            to_delete = [dict(Key=o.key) for o in self.bucket.objects.all()]
            if len(to_delete) > 0:
                log.info("Deleting objects from S3 bucket %s: %s", self.bucket.name, to_delete)
                self.bucket.delete_objects(Delete=dict(
                    Objects=to_delete,
                    Quiet=True
                ))
            log.info("Deleting s3 bucket %s.", self.bucket.name)
            self.bucket.delete()
            log.info("S3 bucket %s was successfully deleted.", self.bucket.name)

    def _upload_resources(self):
        root_key = str_def(rconfig().aws.s3.root_key)
        # TODO: we may want to upload resources to a different path for each run (just in case we run multiple benchmarks in parallel and aws.s3.temporary=False)
        #  for example: root_key+('/'.join(['user', self.uid, name]))
        #  this also requires updating _delete_resources and _ec2_startup_script

        def dest_path(res_path):
            in_input_dir = res_path.startswith(rconfig().input_dir)
            in_user_dir = res_path.startswith(rconfig().user_dir)
            name = os.path.relpath(res_path, start=rconfig().input_dir) if in_input_dir \
                else os.path.relpath(res_path, start=rconfig().user_dir) if in_user_dir \
                else os.path.basename(res_path)
            dest = 'input' if in_input_dir else 'user'
            return root_key+('/'.join([dest, name]))

        upload_paths = [self.benchmark_path] + rconfig().aws.resource_files
        upload_files = list_all_files(upload_paths, rconfig().aws.resource_ignore)
        log.debug("Uploading files to S3: %s", upload_files)
        uploaded_resources = []
        for res in upload_files:
            upload_path = dest_path(res)
            log.info("Uploading `%s` to `%s` on s3 bucket %s.", res, upload_path, self.bucket.name)
            self.bucket.upload_file(res, upload_path)
            uploaded_resources.append(upload_path)
        return uploaded_resources

    def _delete_resources(self):
        if self.uploaded_resources is None:
            return
        # TODO: do we still want to delete resources if concern in _upload_resources is fixed?
        log.info("Deleting uploaded resources `%s` from s3 bucket %s.", self.uploaded_resources, self.bucket.name)
        self.bucket.delete_objects(
            Delete=dict(
                Objects=[dict(Key=res) for res in self.uploaded_resources]
            )
        )

    def _download_results(self, instance_id):
        """
        :param instance_id:
        :return: True iff the main result/scoring file has been successfully downloaded. Other failures are only logged.
        """
        success = False
        instance, ikey = self.instances[instance_id]
        root_key = str_def(rconfig().aws.s3.root_key)

        def download_file(obj, dest, dest_path=None):
            dest_path = dest if dest_path is None else dest_path
            try:
                log.info("Downloading `%s` from s3 bucket %s to `%s`.", obj.key, self.bucket.name, dest_path)
                if isinstance(dest, str):
                    obj.download_file(dest)
                else:
                    obj.download_fileobj(dest)
            except Exception as e:
                log.error("Failed downloading `%s` from s3 bucket %s: %s", obj.key, self.bucket.name, str(e))
                log.exception(e)

        try:
            predictions_objs = [o.Object() for o in self.bucket.objects.filter(Prefix=root_key+('/'.join(['output', ikey, 'predictions'])))]
            scores_objs = [o.Object() for o in self.bucket.objects.filter(Prefix=root_key+('/'.join(['output', ikey, 'scores'])))]
            logs_objs = [o.Object() for o in self.bucket.objects.filter(Prefix=root_key+('/'.join(['output', ikey, 'logs'])))]

            for obj in predictions_objs:
                # it should be safe and good enough to simply save predictions file as usual (after backing up previous prediction)
                dest_path = os.path.join(rconfig().predictions_dir, os.path.basename(obj.key))
                backup_file(dest_path)
                download_file(obj, dest_path)

            for obj in scores_objs:
                basename = os.path.basename(obj.key)
                # FIXME: bypassing the save_scores flag here, do we care?
                board = Scoreboard.from_file(basename)
                if board:
                    with io.BytesIO() as buffer:
                        download_file(obj, buffer, dest_path=board._score_file())
                        with io.TextIOWrapper(io.BytesIO(buffer.getvalue())) as file:
                            df = Scoreboard.load_df(file)
                    df.loc[:,'mode'] = rconfig().run_mode
                    board.append(df).save()
                    success = True
                else:
                    # TODO: test case when there are also backup files in the download
                    dest_path = os.path.join(rconfig().scores_dir, basename)
                    backup_file(dest_path)
                    download_file(obj, dest_path)

            for obj in logs_objs:
                dest_path = os.path.join(rconfig().logs_dir, os.path.basename(obj.key))
                download_file(obj, dest_path)

        except Exception as e:
            log.error("Failed downloading benchmark results from s3 bucket %s: %s", self.bucket.name, str(e))
            log.exception(e)

        return success

    def _create_instance_profile(self):
        """
        see https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/iam-roles-for-amazon-ec2.html
        for steps defined here.
        for restrictions, cf. https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_iam-limits.html
        :return:
        """
        iamc = rconfig().aws.iam
        role_name = iamc.role_name
        profile_name = iamc.instance_profile_name
        if iamc.temporary:
            role_name += ('-' + self.suid)
            profile_name += ('-' + self.suid)

        irole = None
        try:
            self.iam.meta.client.get_role(RoleName=role_name)
            irole = self.iam.Role(role_name)
        except botocore.exceptions.ClientError as e:
            log.info("Role %s doesn't exist, creating it: [%s].", role_name, str(e))

        if not irole:
            ec2_role_trust_policy_json = json.dumps({   # trust role
                'Version': '2012-10-17',  # version of the policy language, cf. https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_elements_version.html
                'Statement': [
                    {
                        'Effect': 'Allow',
                        'Principal': {'Service': 'ec2.amazonaws.com'},
                        'Action': 'sts:AssumeRole'
                    }
                ]
            })
            irole = self.iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=ec2_role_trust_policy_json,
                MaxSessionDuration=iamc.max_role_session_duration_secs
            )
            log.info("Role %s successfully created.", role_name)

        if iamc.s3_policy_name not in [p.name for p in irole.policies.all()]:
            resource_prefix="arn:aws:s3:::{bucket}*/{root_key}".format(bucket=self.bucket.name, root_key=str_def(rconfig().aws.s3.root_key))  # ARN format for s3, cf. https://docs.aws.amazon.com/AmazonS3/latest/dev/s3-arn-format.html
            s3_policy_json = json.dumps({
                'Version': '2012-10-17',
                'Statement': [
                    {
                        'Effect': 'Allow',
                        'Action': 's3:List*',
                        'Resource': 'arn:aws:s3:::{}*'.format(self.bucket.name)
                    },
                    {
                        'Effect': 'Allow',
                        'Action': 's3:GetObject',   # S3 actions, cf. https://docs.aws.amazon.com/AmazonS3/latest/dev/using-with-s3-actions.html
                        'Resource': resource_prefix+'*'
                    },
                    {
                        'Effect': 'Allow',
                        'Action': 's3:PutObject',
                        'Resource': resource_prefix+'output/*'   # technically, we could grant write access for each instance only to its own 'directory', but this is not necessary
                    }
                ]
            })
            self.iam.meta.client.put_role_policy(
                RoleName=irole.name,
                PolicyName=iamc.s3_policy_name,
                PolicyDocument=s3_policy_json
            )

        iprofile = None
        try:
            self.iam.meta.client.get_instance_profile(InstanceProfileName=profile_name)
            iprofile = self.iam.InstanceProfile(profile_name)
        except botocore.exceptions.ClientError as e:
            log.info("Instance profile %s doesn't exist, creating it: [%s].", profile_name, str(e))
        if not iprofile:
            iprofile = self.iam.create_instance_profile(InstanceProfileName=profile_name)
            log.info("Instance profile %s successfully created.", profile_name)
            waiting_time = iamc.credentials_propagation_waiting_time_secs
            steps = math.ceil(waiting_time / 10)
            for i in range(steps):
                log.info("Waiting for new credentials propagation, time left = %ss.", round(waiting_time * (1 - i/steps)))
                time.sleep(waiting_time / steps)

        if irole.name not in [r.name for r in iprofile.roles]:
            iprofile.add_role(RoleName=irole.name)

        return iprofile

    def _delete_iam_entities(self):
        iamc = rconfig().aws.iam
        iprofile = self.instance_profile

        if iprofile is None:
            profile_name = iamc.instance_profile_name
            if iamc.temporary:
                profile_name += ('-' + self.suid)
            try:
                self.iam.meta.client.get_instance_profile(InstanceProfileName=profile_name)
                iprofile = self.iam.InstanceProfile(profile_name)
            except botocore.exceptions.ClientError as e:
                log.info("Instance profile %s doesn't exist, nothing to delete: [%s]", profile_name, str(e))

        if iprofile is not None:
            for role in iprofile.roles:
                log.info("Removing role %s from instance profile %s.", role.name, iprofile.name)
                iprofile.remove_role(RoleName=role.name)
                self._delete_iam_entities_from_role(role.name)
            log.info("Deleting instance profile %s.", iprofile.name)
            iprofile.delete()
            log.info("Instance profile %s was successfully deleted.", iprofile.name)
        else:
            role_name = iamc.role_name
            if iamc.temporary:
                role_name += ('-' + self.suid)
            self._delete_iam_entities_from_role(role_name)

    def _delete_iam_entities_from_role(self, role_name):
        iamc = rconfig().aws.iam
        try:
            self.iam.meta.client.get_role(RoleName=role_name)
            irole = self.iam.Role(role_name)
            for policy in irole.policies.all():
                log.info("Deleting role policy %s from role %s.", policy.name, policy.role_name)
                policy.delete()
                log.info("Policy %s was successfully deleted.", policy.name)
            for profile in irole.instance_profiles.all():
                log.info("Removing instance profile %s from role %s.", profile.name, irole.name)
                profile.remove_role(RoleName=irole.name)
                log.info("Deleting instance profile %s.", profile.name)
                profile.delete()
                log.info("Instance profile %s was successfully deleted.", profile.name)
            log.info("Deleting role %s.", irole.name)
            irole.delete()
            log.info("Role %s was successfully deleted.", irole.name)
        except botocore.exceptions.ClientError as e:
            log.info("Role %s doesn't exist, skipping its deletion: [%s]", iamc.role_name, str(e))

    def _ec2_startup_script(self, instance_key, script_params="", timeout_secs=-1):
        """
        Generates the UserData is cloud-config format for the EC2 instance:
        this script is automatically executed by the instance at the end of its boot process.

        This cloud-config version is currently preferred as the runcmd are always executed sequentially,
        regardless of the previous one raising an error. Especially, the power_state directive is always executed.

        Notes about cloud-config syntax:
            - runcmd: all command are executed sequentially. If one raises an error, the next one is still executed afterwards.
            - power_state:
                * delay (in mn) passed to the shutdown command is executed,
                * timeout (in sec) waiting for cloud-init to complete before triggering shutdown.

        :param instance_key: the unique local identifier for the instance.
            This is different from EC2 instance id as we don't know it yet.
            Mainly used to put output files to dedicated key on s3.
        :param script_params: the custom params passed to the benchmark script, usually only task, fold params
        :return: the UserData for the new ec2 instance
        """
        split_url = rconfig().project_repository.split('#', 2)
        repo = split_url[0]
        branch = 'master' if len(split_url) == 1 else split_url[1]
        cloud_config = """
#cloud-config

package_update: true
package_upgrade: false
packages:
  - python3
  - python3-pip
  - docker.io

runcmd:
  - mkdir -p /s3bucket/input
  - mkdir -p /s3bucket/output
  - mkdir -p /s3bucket/user
  - pip3 install --upgrade awscli
  - aws s3 cp {s3_base_url}input /s3bucket/input --recursive
  - aws s3 cp {s3_base_url}user /s3bucket/user --recursive
  - docker run -v /s3bucket/input:/input -v /s3bucket/output:/output -v /s3bucket/user:/custom --rm {image} {params} -i /input -o /output -u /custom -s skip
  - aws s3 cp /s3bucket/output {s3_base_url}output/{ikey} --recursive
  - rm -f /var/lib/cloud/instances/*/sem/config_scripts_user

final_message: "AutoML benchmark (docker) {ikey} completed after $UPTIME s"

power_state:
  delay: "+1"
  mode: poweroff
  message: "I'm losing power"
  timeout: {timeout}
  condition: True
""" if rconfig().aws.use_docker else """
#cloud-config

package_update: true
package_upgrade: false
packages:
  - curl
  - wget
  - unzip
  - git
  - python3
  - python3-pip
  - python3-venv

runcmd:
  - pip3 install --upgrade awscli
  - python3 -m venv /venvs/bench
  - alias PIP='/venvs/bench/bin/pip3'
  - alias PY='/venvs/bench/bin/python3 -W ignore'
  - alias PIP_REQ='xargs -L 1 /venvs/bench/bin/pip3 install --no-cache-dir'
  - mkdir -p /s3bucket/input
  - mkdir -p /s3bucket/output
  - mkdir -p /s3bucket/user
  - mkdir /repo
  - cd /repo
  - git clone --depth 1 --single-branch --branch {branch} {repo} .
  - PIP install --upgrade pip=={pip_version}
  - PIP_REQ < requirements.txt
#  - until aws s3 ls {s3_base_url}; do echo "waiting for credentials"; sleep 10; done
  - aws s3 cp {s3_base_url}input /s3bucket/input --recursive
  - aws s3 cp {s3_base_url}user /s3bucket/user --recursive
  - PY {script} {params} -i /s3bucket/input -o /s3bucket/output -u /s3bucket/user -s only 
  - PY {script} {params} -i /s3bucket/input -o /s3bucket/output -u /s3bucket/user
  - aws s3 cp /s3bucket/output {s3_base_url}output/{ikey} --recursive
  - rm -f /var/lib/cloud/instances/*/sem/config_scripts_user

final_message: "AutoML benchmark {ikey} completed after $UPTIME s"

power_state:
  delay: "+1"
  mode: poweroff
  message: "I'm losing power"
  timeout: {timeout}
  condition: True
"""
        return cloud_config.format(
            repo=repo,
            branch=branch,
            image=DockerBenchmark.docker_image_name(self.framework_def),
            pip_version=rconfig().versions.pip,
            s3_base_url="s3://{bucket}/{root}".format(bucket=self.bucket.name, root=str_def(rconfig().aws.s3.root_key)),
            script=rconfig().script,
            ikey=instance_key,
            params=script_params,
            timeout=timeout_secs if timeout_secs > 0 else rconfig().aws.max_timeout_seconds,
        )

    def _ec2_startup_script_bash(self, instance_key, script_params="", timeout_secs=-1):
        """
        Backup UserData version if the cloud-config version doesn't work as expected.

        Generates the UserData is bash format for the EC2 instance:
        this script is automatically executed by the instance at the end of its boot process.
        TODO: current version doesn't handle errors at all, that's why the cloud-config version is currently preferred.
        :param instance_key: the unique local identifier for the instance.
            This is different from EC2 instance id as we don't know it yet.
            Mainly used to put output files to dedicated key on s3.
        :param script_params: the custom params passed to the benchmark script, usually only task, fold params
        :return: the UserData for the new ec2 instance
        """
        split_url = rconfig().project_repository.split('#', 2)
        repo = split_url[0]
        branch = 'master' if len(split_url) == 1 else split_url[1]
        return """#!/bin/bash
apt-get update
#apt-get -y upgrade
apt-get install -y curl wget unzip git
apt-get install -y python3 python3-pip python3-venv
#apt-get install -y docker.io

pip3 install --upgrade awscli
python3 -m venv /venvs/bench
alias PIP='/venvs/bench/bin/pip3'
alias PY='/venvs/bench/bin/python3 -W ignore'

mkdir -p /s3bucket/input
mkdir -p /s3bucket/output
mkdir -p /s3bucket/user
mkdir ~/repo
cd ~/repo
git clone --depth 1 --single-branch --branch {branch} {repo} .

PIP install --upgrade pip=={pip_version}
xargs -L 1 PIP install --no-cache-dir < requirements.txt
PIP install --upgrade awscli

aws s3 cp {s3_base_url}input /s3bucket/input --recursive
aws s3 cp {s3_base_url}user /s3bucket/user --recursive
PY {script} {params} -i /s3bucket/input -o /s3bucket/output -u /s3bucket/user -s only 
PY {script} {params} -i /s3bucket/input -o /s3bucket/output -u /s3bucket/user
aws s3 cp /s3bucket/output {s3_base_url}output/{ikey} --recursive
rm -f /var/lib/cloud/instances/*/sem/config_scripts_user
shutdown -P +1 "I'm losing power"
""".format(
            repo=repo,
            branch=branch,
            pip_version=rconfig().versions.pip,
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

