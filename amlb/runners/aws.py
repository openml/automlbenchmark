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
from concurrent.futures import ThreadPoolExecutor
import copy as cp
import datetime as dt
import itertools
import json
import logging
import math
import operator as op
import os
from posixpath import join as url_join, relpath as url_relpath
import re
import time
import threading
from urllib.parse import quote_plus as uenc

import boto3
import botocore.exceptions

from ..benchmark import Benchmark, SetupMode
from ..datautils import read_csv, write_csv
from ..job import Job
from ..resources import config as rconfig, get as rget
from ..results import ErrorResult, Scoreboard, TaskResult
from ..utils import Namespace as ns, datetime_iso, file_filter, flatten, list_all_files, normalize_path, str_def, tail, touch
from .docker import DockerBenchmark


log = logging.getLogger(__name__)


class AWSBenchmark(Benchmark):
    """AWSBenchmark
    an extension of Benchmark class, to run benchmarks on AWS
    """

    @classmethod
    def fetch_results(cls, instances_file, instance_selector=None):
        bench = cls(None, None, None)
        bench._load_instances(normalize_path(instances_file))
        inst = next(inst for inst in bench.instances.values())
        bench.sid = inst.session
        bucket_name = re.match(r's3://([\w\-.]+)/.*', inst.s3_dir).group(1)
        bench.s3 = boto3.resource('s3', region_name=bench.region)
        bench.bucket = bench._create_s3_bucket(bucket_name, auto_create=False)
        instance_selector = (lambda *_: True) if instance_selector is None else instance_selector
        for iid, _ in filter(instance_selector, bench.instances.items()):
            bench._download_results(iid)

    @classmethod
    def reconnect(cls, instances_file):
        bench = cls(None, None, None)
        bench._load_instances(normalize_path(instances_file))
        inst = next(inst for inst in bench.instances.values())
        bench.sid = inst.session
        bench.setup(SetupMode.script)
        bench._exec_start()
        bench._monitoring_start()

        def to_job(iid, inst):
            inst.instance = bench.ec2.Instance(iid)
            job = Job(inst.key)
            job.instance_id = iid

            def _run(job_self):
                return bench._wait_for_results(job_self)

            def _on_done(job_self):
                terminate = bench._download_results(job_self.instance_id)
                if not terminate and rconfig().aws.ec2.terminate_instances == 'success':
                    log.warning("[WARNING]: EC2 Instance %s won't be terminated as we couldn't download the results: "
                        "please terminate it manually or restart it (after clearing its UserData) if you want to inspect the instance.",
                        job_self.instance_id)
                bench._stop_instance(job_self.instance_id, terminate=terminate)

            job._run = _run.__get__(job)
            job._on_done = _on_done.__get__(job)

        jobs = list(itertools.starmap(to_job, bench.instances.items()))
        bench.parallel_jobs = len(jobs)
        try:
            bench._run_jobs(jobs)
        finally:
            bench.cleanup()

    def __init__(self, framework_name, benchmark_name, constraint_name, region=None):
        """

        :param framework_name:
        :param benchmark_name:
        :param constraint_name:
        :param region:
        """
        super().__init__(framework_name, benchmark_name, constraint_name)
        self.suid = datetime_iso(micros=True, no_sep=True)  # short sid for AWS entities whose name length is limited
        self.region = region if region \
            else rconfig().aws.region if rconfig().aws['region'] \
            else boto3.session.Session().region_name
        self.ami = rconfig().aws.ec2.regions[self.region].ami
        self.cloudwatch = None
        self.ec2 = None
        self.iam = None
        self.s3 = None
        self.bucket = None
        self.uploaded_resources = None
        self.instance_profile = None
        self.instances = {}
        self.jobs = []
        self.exec = None
        self.monitoring = None
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
        if mode == SetupMode.skip:
            log.warning("AWS setup mode set to unsupported {mode}, ignoring.".format(mode=mode))
        # S3 setup to exchange files between local and ec2 instances
        self.s3 = boto3.resource('s3', region_name=self.region)
        self.bucket = self._create_s3_bucket()
        self.uploaded_resources = self._upload_resources() if mode != SetupMode.script else []

        # IAM setup to secure exchanges between s3 and ec2 instances
        self.iam = boto3.resource('iam', region_name=self.region)
        if mode == SetupMode.force:
            log.warning("Cleaning up previously created IAM entities if any.")
            self._delete_iam_entities()
        self.instance_profile = self._create_instance_profile()

        # EC2 setup to prepare creation of ec2 instances
        self.ec2 = boto3.resource('ec2', region_name=self.region)
        self.cloudwatch = boto3.resource('cloudwatch', region_name=self.region)

    def cleanup(self):
        self._stop_all_instances()
        self._monitoring_stop()
        self._exec_stop()
        if rconfig().aws.s3.delete_resources is True:
            self._delete_resources()
        if rconfig().aws.iam.temporary is True:
            self._delete_iam_entities()
        if rconfig().aws.s3.temporary is True:
            self._delete_s3_bucket()

    def run(self, task_name=None, fold=None):
        task_defs = self._get_task_defs(task_name)  # validates tasks
        self._exec_start()
        self._monitoring_start()
        if self.parallel_jobs > 1:
            if rconfig().aws.minimize_instances:
                # use one instance per task: all folds executed on same instance
                try:
                    jobs = flatten([self._make_aws_job([task_def.name], fold) for task_def in task_defs])
                    results = self._run_jobs(jobs)
                    return self._process_results(results, task_name=task_name)
                finally:
                    self.cleanup()
            else:
                # use one instance per fold per task
                return super().run(task_name, fold)
        else:
            # use one instance for all
            try:
                task_names = None if task_name is None else [task_def.name for task_def in task_defs]
                job = self._make_aws_job(task_names, fold)
                results = self._run_jobs([job])
                return self._process_results(results, task_name=task_name)
            finally:
                self.cleanup()

    def _make_job(self, task_def, fold=int):
        return self._make_aws_job([task_def.name], [fold])

    def _exec_start(self):
        if self.exec is not None:
            return
        self.exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aws_exec_")

    def _exec_stop(self):
        if self.exec is None:
            return
        try:
            self.exec.shutdown(wait=True)
        except:
            pass
        finally:
            self.exec = None

    def _exec_send(self, fn, *args, **kwargs):
        if self.exec is not None:
            self.exec.submit(fn, *args, **kwargs)
        else:
            log.warning("Sending exec function while executor is not started: executing the function in the calling thread.")
            try:
                fn(*args, **kwargs)
            except:
                pass

    def _make_aws_job(self, task_names=None, folds=None):
        task_names = [] if task_names is None else task_names
        folds = [] if folds is None else [str(f) for f in folds]
        task_def = (self._get_task_def(task_names[0]) if len(task_names) >= 1
                    else self._get_task_def('__defaults__', include_disabled=True, fail_on_missing=False) or ns(name='all'))
        task_def = cp.copy(task_def)
        tconfig = rconfig()['t'] or ns()  # handle task params from cli (-Xt.foo=bar)
        for k, v in tconfig:
            setattr(task_def, k, v)

        instance_def = ns()
        instance_def.type = (task_def.ec2_instance_type if 'ec2_instance_type' in task_def
                             else '.'.join([rconfig().aws.ec2.instance_type.series, rconfig().aws.ec2.instance_type.map.default]))
        instance_def.volume_type = (task_def.ec2_volume_type if 'ec2_volume_type' in task_def
                                    else rconfig().aws.ec2.volume_type)
        instance_def.volume_size = (math.ceil((task_def.min_vol_size_mb + rconfig().benchmarks.os_vol_size_mb) / 1024.) if task_def.min_vol_size_mb > 0
                                    else None)

        timeout_secs = (task_def.max_runtime_seconds if 'max_runtime_seconds' in task_def
                        else sum([task.max_runtime_seconds for task in self.benchmark_def]))
        timeout_secs += rconfig().aws.overhead_time_seconds

        job = Job(rconfig().token_separator.join([
            'aws',
            self.benchmark_name,
            self.constraint_name,
            ' '.join(task_names) if len(task_names) > 0 else 'all',
            ' '.join(folds),
            self.framework_name
        ]))
        job.instance_id = None

        def _run(job_self):
            resources_root = "/custom" if rconfig().aws.use_docker else "/s3bucket/user"
            job_self.instance_id = self._start_instance(
                instance_def,
                script_params="{framework} {benchmark} {constraint} {task_param} {folds_param} -Xseed={seed}".format(
                    framework=self.framework_name,
                    benchmark=(self.benchmark_name if self.benchmark_path is None or self.benchmark_path.startswith(rconfig().root_dir)
                               else "{}/{}".format(resources_root, self._rel_path(self.benchmark_path))),
                    constraint=self.constraint_name,
                    task_param='' if len(task_names) == 0 else ' '.join(['-t']+task_names),
                    folds_param='' if len(folds) == 0 else ' '.join(['-f']+folds),
                    seed=rget().seed(int(folds[0])) if len(folds) == 1 else rconfig().seed,
                ),
                # instance_key='_'.join([job.name, datetime_iso(micros=True, time_sep='.')]),
                instance_key=job.name,
                timeout_secs=timeout_secs
            )
            try:
                return self._wait_for_results(job_self)
            except Exception as e:
                fold = int(folds[0]) if len(folds) > 0 else -1
                results = TaskResult(task_def=task_def, fold=fold, constraint=self.constraint_name)
                return results.compute_scores(self.framework_name, [], result=ErrorResult(e))

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
        instance = self.instances[job.instance_id].instance
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

        interrupt = threading.Event()
        while not interrupt.is_set():
            inst_desc = self.instances[job.instance_id] if job.instance_id in self.instances else ns()
            if inst_desc['abort']:
                self._update_instance(job.instance_id, status='aborted')
                raise Exception("Aborting instance {} for job {}.".format(job.instance_id, job.name))
            try:
                state = instance.state['Name']
                state_code = instance.state['Code']
                log.info("[%s] checking job %s on instance %s: %s [%s].", datetime_iso(), job.name, job.instance_id, state, state_code)
                log_console()
                self._update_instance(job.instance_id, status=state)

                if state_code == 16:
                    if inst_desc['meta_info'] is None:
                        meta_info = dict(
                            instance_type=instance.instance_type,
                            launch_time=str(instance.launch_time),
                            public_dns_name=instance.public_dns_name,
                            public_ip=instance.public_ip_address,
                            private_dns_name=instance.private_dns_name,
                            private_ip=instance.private_ip_address,
                            availability_zone=instance.placement['AvailabilityZone'],
                            subnet_id=instance.subnet_id,
                        )
                        self._update_instance(job.instance_id, meta_info=meta_info)
                        log.info("Running EC2 instance %s: %s", instance.id, meta_info)
                elif state_code > 16:     # ended instance
                    log.info("EC2 instance %s is %s: %s", job.instance_id, state, instance.state_reason['Message'])
                    interrupt.set()
            except Exception as e:
                log.exception(e)
            finally:
                interrupt.wait(rconfig().aws.query_frequency_seconds)

    def _get_cpu_activity(self, iid, delta_minutes=60, period_minutes=5):
        now = dt.datetime.utcnow()
        resp = self.cloudwatch.meta.client.get_metric_statistics(
            Namespace='AWS/EC2',
            MetricName='CPUUtilization',
            Dimensions=[dict(Name='InstanceId', Value=iid)],
            StartTime=now - dt.timedelta(minutes=delta_minutes),
            EndTime=now,
            Period=60*period_minutes,
            Statistics=['Average'],
            Unit='Percent'
        )
        return [activity['Average'] for activity in sorted(resp['Datapoints'], key=op.itemgetter('Timestamp'), reverse=True)]

    def _is_hanging(self, iid):
        cpu_config = rconfig().aws.ec2.monitoring.cpu
        activity = self._get_cpu_activity(iid,
                                          delta_minutes=cpu_config.delta_minutes,
                                          period_minutes=cpu_config.period_minutes)
        threshold = cpu_config.threshold
        min_activity_len = int(cpu_config.delta_minutes / cpu_config.period_minutes)
        return len(activity) >= min_activity_len and all([a < threshold for a in activity])

    def _monitoring_start(self):
        if self.monitoring is not None:
            return

        interrupt = threading.Event()

        def cpu_monitor():
            cpu_config = rconfig().aws.ec2.monitoring.cpu
            if cpu_config.query_frequency_seconds <= 0:
                return
            while not interrupt.is_set():
                try:
                    hanging_instances = list(filter(self._is_hanging, self.instances.keys()))
                    for inst in hanging_instances:
                        if inst in self.instances:
                            inst_desc = self.instances[inst]
                            log.warning("WARN: Instance %s (%s) has no CPU activity in the last %s minutes.", inst, inst_desc.key, cpu_config.delta_minutes)
                            if cpu_config.abort_inactive_instances:
                                inst_desc.abort = True
                except Exception as e:
                    log.exception(e)
                finally:
                    interrupt.wait(cpu_config.query_frequency_seconds)

        self.monitoring = ns(executor=ThreadPoolExecutor(max_workers=1, thread_name_prefix="aws_monitoring_"),
                             interrupt=interrupt)
        self.monitoring.executor.submit(cpu_monitor)

    def _monitoring_stop(self):
        if self.monitoring is None:
            return
        try:
            self.monitoring.interrupt.set()
            self.monitoring.executor.shutdown(wait=False)
        except:
            pass
        finally:
            self.monitoring = None

    def _start_instance(self, instance_def, script_params="", instance_key=None, timeout_secs=-1):
        log.info("Starting new EC2 instance with params: %s.", script_params)
        inst_key = instance_key.lower() if instance_key \
            else "{}_p{}_i{}".format(self.sid,
                                     re.sub(r"[\s-]", '', script_params),
                                     datetime_iso(micros=True, time_sep='.')).lower()
        # TODO: don't know if it would be considerably faster to reuse previously stopped instances sometimes
        #   instead of always creating a new one:
        #   would still need to set a new UserData though before restarting the instance.
        ec2_config = rconfig().aws.ec2
        try:
            ebs = dict(VolumeType=instance_def.volume_type)
            if instance_def.volume_size:
                ebs['VolumeSize'] = instance_def.volume_size

            instance_params = dict(
                BlockDeviceMappings=[dict(
                    DeviceName=ec2_config.root_device_name,
                    Ebs=ebs
                )],
                IamInstanceProfile=dict(Name=self.instance_profile.name),
                ImageId=self.ami,
                InstanceType=instance_def.type,
                MinCount=1,
                MaxCount=1,
                SubnetId=ec2_config.subnet_id,
                TagSpecifications=[
                    dict(
                        ResourceType='instance',
                        Tags=[
                            dict(Key='Name', Value=f"benchmark_{inst_key}")
                        ]
                    ),
                    dict(
                        ResourceType='volume',
                        Tags=[
                            dict(Key='Name', Value=f"benchmark_{inst_key}")
                        ]
                    ),
                ],
                UserData=self._ec2_startup_script(inst_key, script_params=script_params, timeout_secs=timeout_secs)
            )
            if ec2_config.key_name is not None:
                instance_params.update(KeyName=ec2_config.key_name)
            if ec2_config.security_groups:
                instance_params.update(SecurityGroups=ec2_config.security_groups)

            instance = self.ec2.create_instances(**instance_params)[0]
            log.info("Started EC2 instance %s", instance.id)
            self.instances[instance.id] = ns(instance=instance, key=inst_key, status='started', success='',
                                             start_time=datetime_iso(), stop_time='',
                                             meta_info=None)
        except Exception as e:
            fake_iid = "no_instance_{}".format(len(self.instances)+1)
            self.instances[fake_iid] = ns(instance=None, key=inst_key, status='failed', success=False,
                                          start_time=datetime_iso(), stop_time=datetime_iso(),
                                          meta_info=None)
            raise e
        finally:
            self._exec_send(self._save_instances)
        return instance.id

    def _stop_instance(self, instance_id, terminate=None):
        instance = self.instances[instance_id].instance
        self.instances[instance_id].instance = None
        if instance is None:
            return

        terminate_config = rconfig().aws.ec2.terminate_instances
        if terminate_config in ['always', True]:
            terminate = True
        elif terminate_config in ['never', False]:
            terminate = False
        else:
            terminate = False if terminate is None else terminate

        try:
            log.info("%s EC2 instances %s.", "Terminating" if terminate else "Stopping", instance_id)
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
                state = response['TerminatingInstances'][0]['CurrentState']['Name']
                log.info("Instance %s state: %s.", instance_id, state)
                self._update_instance(instance_id, status=state, stop_time=datetime_iso())
            except:
                pass

    def _update_instance(self, instance_id, **kwargs):
        do_save = False
        inst = self.instances[instance_id]
        for k, v in kwargs.items():
            if k in inst and inst[k] != v:
                inst[k] = v
                do_save = True
        if do_save:
            self._exec_send(lambda: self._save_instances())

    def _stop_all_instances(self):
        for iid in self.instances.keys():
            self._stop_instance(iid)

    def _save_instances(self):
        write_csv([(iid,
                    self.instances[iid].status,
                    self.instances[iid].success,
                    self.instances[iid].start_time,
                    self.instances[iid].stop_time,
                    self.sid,
                    self.instances[iid].key,
                    self._s3_key(self.sid, instance_key_or_id=iid, absolute=True),
                    self.instances[iid].meta_info
                    ) for iid in self.instances.keys()],
                  columns=['ec2', 'status', 'success', 'start_time', 'stop_time', 'session', 'instance_key', 's3_dir', 'meta_info'],
                  path=os.path.join(self.output_dirs.session, 'instances.csv'))

    def _load_instances(self, instances_file):
        df = read_csv(instances_file)
        self.instances = {row['ec2']: ns(
            status=row['status'],
            success=row['success'],
            session=row['session'],
            key=row['instance_key'],
            s3_dir=row['s3_dir'],
        ) for idx, row in df.iterrows()}

    def _s3_key(self, main_dir, *subdirs, instance_key_or_id=None, absolute=False, encode=False):
        root_key = str_def(rconfig().aws.s3.root_key)
        if instance_key_or_id is None:
            ikey = ''
        elif instance_key_or_id in self.instances.keys():
            ikey = self.instances[instance_key_or_id].key
        else:
            ikey = instance_key_or_id
        tokens = [main_dir, ikey, *subdirs]
        if encode:
            tokens = map(uenc, tokens)
        rel_key = url_join(root_key, *tokens)
        return url_join('s3://', self.bucket.name, rel_key) if absolute else rel_key

    def _s3_session(self, *subdirs, **kwargs):
        return self._s3_key(self.sid, *subdirs, **kwargs)

    def _s3_user(self, *subdirs, **kwargs):
        return self._s3_key(self.sid, 'user', *subdirs, **kwargs)

    def _s3_input(self, *subdirs, **kwargs):
        return self._s3_key(self.sid, 'input', *subdirs, **kwargs)

    def _s3_output(self, instance_key_or_id, *subdirs, **kwargs):
        return self._s3_key(self.sid, 'output', *subdirs, instance_key_or_id=instance_key_or_id, **kwargs)

    def _create_s3_bucket(self, bucket_name=None, auto_create=True):
        # cf. s3 restrictions: https://docs.aws.amazon.com/AmazonS3/latest/dev/BucketRestrictions.html
        if bucket_name is None:
            bucket_name = rconfig().aws.s3.bucket
            if rconfig().aws.s3.temporary:
                bucket_name += ('-' + self.suid)
        try:
            self.s3.meta.client.head_bucket(Bucket=bucket_name)
            bucket = self.s3.Bucket(bucket_name)
        except botocore.exceptions.ClientError as e:
            error_code = int(e.response['Error']['Code'])
            if error_code == 404 and auto_create:
                log.info("%s bucket doesn't exist, creating it in region %s.", bucket_name, self.region)
                bucket = self.s3.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration=dict(
                        LocationConstraint=self.region
                    )
                )
                log.info("S3 bucket %s was successfully created.", bucket_name)
            else:
                if error_code == 403:
                    log.error("You don't have access rights to S3 bucket %s.\n"
                              "Please ensure that you specified a unique `aws.s3.bucket` in your config file"
                              " or verify that your AWS account is correctly configured"
                              " (cf. docs/README.md for more details).", bucket_name)
                elif error_code == 404:
                    log.error("S3 bucket %s does not exist and auto-creation is disabled.", bucket_name)
                raise e
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

    def _rel_path(self, res_path):
        in_app_dir = res_path.startswith(rconfig().root_dir)
        if in_app_dir:
            return None
        in_input_dir = res_path.startswith(rconfig().input_dir)
        in_user_dir = res_path.startswith(rconfig().user_dir)
        return (os.path.relpath(res_path, start=rconfig().input_dir) if in_input_dir
                else os.path.relpath(res_path, start=rconfig().user_dir) if in_user_dir
                else os.path.basename(res_path))

    def _dest_path(self, res_path):
        name = self._rel_path(res_path)
        if name is None:
            return None
        in_input_dir = res_path.startswith(rconfig().input_dir)
        return self._s3_input(name) if in_input_dir else self._s3_user(name)

    def _upload_resources(self):
        default_paths = [self.benchmark_path] if self.benchmark_path is not None else []
        upload_paths = default_paths + rconfig().aws.resource_files
        upload_files = list_all_files(upload_paths, file_filter(exclude=rconfig().aws.resource_ignore))
        log.debug("Uploading files to S3: %s", upload_files)
        uploaded_resources = []
        for res in upload_files:
            upload_path = self._dest_path(res)
            if upload_path is None:
                log.debug("Skipping upload of `%s` to s3 bucket.", res)
                continue
            log.info("Uploading `%s` to `%s` on s3 bucket %s.", res, upload_path, self.bucket.name)
            self.bucket.upload_file(res, upload_path)
            uploaded_resources.append(upload_path)
        return uploaded_resources

    def _delete_resources(self):
        if self.uploaded_resources is None:
            return
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
        def download_file(obj, dest, dest_display_path=None):
            dest_display_path = dest if dest_display_path is None else dest_display_path
            try:
                log.info("Downloading `%s` from s3 bucket %s to `%s`.", obj.key, self.bucket.name, dest_display_path)
                if isinstance(dest, str):
                    touch(dest)
                    obj.download_file(dest)
                else:
                    obj.download_fileobj(dest)
            except Exception as e:
                log.error("Failed downloading `%s` from s3 bucket %s: %s", obj.key, self.bucket.name, str(e))
                log.exception(e)

        success = self.instances[instance_id].success is True
        try:
            instance_output_key = self._s3_output(instance_id, encode=True)
            objs = [o.Object() for o in self.bucket.objects.filter(Prefix=instance_output_key)]
            session_key = self._s3_session(encode=True)
            # result_key = self._s3_output(instance_id, Scoreboard.results_file, encode=True)
            for obj in objs:
                rel_path = url_relpath(obj.key, start=session_key)
                dest_path = os.path.join(self.output_dirs.session, rel_path)
                download_file(obj, dest_path)
                # if obj.key == result_key:
                if not success and os.path.basename(obj.key) == Scoreboard.results_file:
                    if rconfig().results.save:
                        self._exec_send(lambda path: self._append(Scoreboard.load_df(path)), dest_path)
                    success = True
        except Exception as e:
            log.error("Failed downloading benchmark results from s3 bucket %s: %s", self.bucket.name, str(e))
            log.exception(e)

        log.info("Instance `%s` success=%s", instance_id, success)
        self._update_instance(instance_id, success=success)
        return success

    def _create_instance_profile(self):
        """
        see https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/iam-roles-for-amazon-ec2.html
        for steps defined here.
        for restrictions, cf. https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_iam-limits.html
        :return:
        """
        s3c = rconfig().aws.s3
        iamc = rconfig().aws.iam
        bucket_prefix = (s3c.bucket+'-') if (s3c.temporary and not iamc.temporary) else self.bucket.name
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
            resource_prefix="arn:aws:s3:::{bucket}*/{root_key}".format(bucket=bucket_prefix, root_key=str_def(s3c.root_key))  # ARN format for s3, cf. https://docs.aws.amazon.com/AmazonS3/latest/dev/s3-arn-format.html
            s3_policy_json = json.dumps({
                'Version': '2012-10-17',
                'Statement': [
                    {
                        'Effect': 'Allow',
                        'Action': 's3:List*',
                        'Resource': 'arn:aws:s3:::{}*'.format(bucket_prefix)
                    },
                    {
                        'Effect': 'Allow',
                        'Action': 's3:GetObject',   # S3 actions, cf. https://docs.aws.amazon.com/AmazonS3/latest/dev/using-with-s3-actions.html
                        'Resource': resource_prefix+'*'
                    },
                    {
                        'Effect': 'Allow',
                        'Action': 's3:PutObject',
                        'Resource': resource_prefix+'*' # technically, we could grant write access for each instance only to its own 'directory', but this is not necessary
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
        script_extra_params = "--session="
        cloud_config = """
#cloud-config

package_update: true
package_upgrade: false
packages:
  - python3
  - python3-pip
  - docker.io

runcmd:
  - apt-get -y remove unattended-upgrades
  - systemctl stop apt-daily.timer
  - systemctl disable apt-daily.timer
  - systemctl disable apt-daily.service
  - systemctl daemon-reload
  - mkdir -p /s3bucket/input
  - mkdir -p /s3bucket/output
  - mkdir -p /s3bucket/user
  - pip3 install -U awscli wheel
  - aws s3 cp '{s3_input}' /s3bucket/input --recursive
  - aws s3 cp '{s3_user}' /s3bucket/user --recursive
  - docker run {docker_options} -v /s3bucket/input:/input -v /s3bucket/output:/output -v /s3bucket/user:/custom --rm {image} {params} -i /input -o /output -u /custom -s skip -Xrun_mode=aws.docker {extra_params}
  - aws s3 cp /s3bucket/output '{s3_output}' --recursive
  #- rm -f /var/lib/cloud/instance/sem/config_scripts_user

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
  - apt-get -y remove unattended-upgrades
  - systemctl stop apt-daily.timer
  - systemctl disable apt-daily.timer
  - systemctl disable apt-daily.service
  - systemctl daemon-reload
  - pip3 install -U awscli wheel
  - mkdir -p /s3bucket/input
  - mkdir -p /s3bucket/output
  - mkdir -p /s3bucket/user
  - mkdir /repo
  - cd /repo
  - git clone --depth 1 --single-branch --branch {branch} {repo} .
  - python3 -m venv venv
  - alias PIP='/repo/venv/bin/pip3'
  - alias PY='/repo/venv/bin/python3 -W ignore'
  - alias PIP_REQ='xargs -L 1 /repo/venv/bin/pip3 install --no-cache-dir'
#  - PIP install -U pip=={pip_version}
  - PIP install -U pip
  - PIP_REQ < requirements.txt
#  - until aws s3 ls '{s3_base_url}'; do echo "waiting for credentials"; sleep 10; done
  - aws s3 cp '{s3_input}' /s3bucket/input --recursive
  - aws s3 cp '{s3_user}' /s3bucket/user --recursive
  - PY {script} {params} -i /s3bucket/input -o /s3bucket/output -u /s3bucket/user -s only --session=
  - PY {script} {params} -i /s3bucket/input -o /s3bucket/output -u /s3bucket/user -Xrun_mode=aws -Xproject_repository={repo}#{branch} {extra_params}
  - aws s3 cp /s3bucket/output '{s3_output}' --recursive
#  - rm -f /var/lib/cloud/instance/sem/config_scripts_user

final_message: "AutoML benchmark {ikey} completed after $UPTIME s"

power_state:
  delay: "+1"
  mode: poweroff
  message: "I'm losing power"
  timeout: {timeout}
  condition: True
"""
        return cloud_config.format(
            repo=rget().project_info.repo,
            branch=rget().project_info.branch,
            image=rconfig().docker.image or DockerBenchmark.image_name(self.framework_def),
            pip_version=rconfig().versions.pip,
            s3_base_url=self._s3_session(absolute=True, encode=True),
            s3_user=self._s3_user(absolute=True, encode=True),
            s3_input=self._s3_input(absolute=True, encode=True),
            s3_output=self._s3_output(instance_key, absolute=True, encode=True),
            script=rconfig().script,
            ikey=instance_key,
            params=script_params,
            extra_params=script_extra_params,
            docker_options=rconfig().docker.run_extra_options,
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
        script_extra_params = "--session="
        return """#!/bin/bash
apt-get update
#apt-get -y upgrade
apt-get -y install curl wget unzip git
apt-get -y install python3 python3-pip python3-venv
#apt-get -y install docker.io

pip3 install -U awscli wheel

mkdir -p /s3bucket/input
mkdir -p /s3bucket/output
mkdir -p /s3bucket/user
mkdir /repo
cd /repo
git clone --depth 1 --single-branch --branch {branch} {repo} .

python3 -m venv venv
alias PIP='/repo/venv/bin/pip3'
alias PY='/repo/venv/bin/python3 -W ignore'
#PIP install -U pip=={pip_version}
PIP install -U pip wheel
xargs -L 1 PIP install --no-cache-dir < requirements.txt

aws s3 cp '{s3_input}' /s3bucket/input --recursive
aws s3 cp '{s3_user}' /s3bucket/user --recursive
PY {script} {params} -i /s3bucket/input -o /s3bucket/output -u /s3bucket/user -s only --session=
PY {script} {params} -i /s3bucket/input -o /s3bucket/output -u /s3bucket/user -Xrun_mode=aws -Xproject_repository={repo}#{branch} {extra_params}
aws s3 cp /s3bucket/output '{s3_output}' --recursive
#rm -f /var/lib/cloud/instance/sem/config_scripts_user
shutdown -P +1 "I'm losing power"
""".format(
            repo=rget().project_info.repo,
            branch=rget().project_info.branch,
            pip_version=rconfig().versions.pip,
            s3_base_url=self._s3_session(absolute=True, encode=True),
            s3_user=self._s3_user(absolute=True, encode=True),
            s3_input=self._s3_input(absolute=True, encode=True),
            s3_output=self._s3_output(instance_key, absolute=True, encode=True),
            script=rconfig().script,
            ikey=instance_key,
            params=script_params,
            extra_params=script_extra_params,
        )


class AWSRemoteBenchmark(Benchmark):

    # TODO: idea is to handle results progressively on the remote side and push results as soon as they're generated
    #   this would allow to safely run multiple tasks on single AWS instance

    def __init__(self, framework_name, benchmark_name, constraint_name, region=None):
        self.region = region
        self.s3 = boto3.resource('s3', region_name=self.region)
        self.bucket = self._init_bucket()
        self._download_resources()
        super().__init__(framework_name, benchmark_name, constraint_name)

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

