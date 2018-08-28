#!/usr/bin/python3

import boto3
import re

class AwsDockerOMLRun:

  setup = '#!/bin/bash\ndocker run --rm'
  token = "6744dfceeb4d2b4a9e60874bcd46b3a1"

  def __init__(self, aws_instance_type, aws_instance_image, docker_image, openml_id, fold, runtime, cores, metric, region_name=None):
    self.aws_instance_type = aws_instance_type
    self.aws_instance_image = aws_instance_image
    self.docker_image = docker_image
    self.openml_id = openml_id
    self.fold = fold
    self.runtime = runtime
    self.cores = cores
    self.metric = metric
    region_name = region_name if region_name is not None else boto3.session.Session().region_name
    self.ec2_resource = boto3.resource("ec2", region_name=region_name)  # Maybe this should be a class variable, not sure
    self.instance = None

  def createInstanceRun(self):
    setup = "%s %s -f %i -t %i -s %i -p %i -m %s" % (self.setup, self.docker_image, self.fold, self.openml_id, self.runtime, self.cores, self.metric)
    if self.instance is not None:
      print("Instance already exists, terminate existing instance")
      self.terminateInstance()

    self.instance = self.ec2_resource.create_instances(
      ImageId=self.aws_instance_image,
      MinCount=1,
      MaxCount=1,
      InstanceType=self.aws_instance_type,
      UserData=setup)[0]

  def terminateInstance(self):
    if self.instance is not None:
      termination_result = self.instance.terminate()
      if not termination_result["TerminatingInstances"][0]["CurrentState"]["Code"] == 32:
        print("Instance could not be terminated!")
      else:
        print("Termination successful")

  def getResult(self):

    if self.instance is None:
      print("No instance created, run createInstanceRun first")
    self.instance.load()

    if not self.instance.state["Name"] == "running":
      print("Instance %s" % (self.instance.state["Name"]))
    else:
      raw_log = self.instance.console_output(Latest = True)
      if "Output" in raw_log.keys():
        out = raw_log["Output"].splitlines()
        out = [x for x in out if re.search(self.token, x)]
        if len(out) == 1:
          return {"res":out[0].split(" ")[-1], "log":raw_log}
        #else:
        #  print("Run finished without result!")
        #  return {"res":float('nan'), "log":raw_log}
if __name__ == "main":

  from time import sleep
  from os import popen

  instance = "m5.xlarge" # instance type
  image = "ami-0615f1e34f8d36362" # aws instance image
  dockerImage = "jnkthms/tpot" # docker image
  openmlid = 59
  runtime = 600
  cores = 4
  run = AwsDockerOMLRun(aws_instance_type = instance,
    aws_instance_image = image, docker_image = dockerImage, openml_id = openmlid, fold = 1,
    runtime = runtime, cores = cores, metric = "acc")

  run.createInstanceRun()
  res = []
  for i in range(100):
    print(i)
    sleep(10)
    run.getResult()

  r = run.instance.console_output()

  run.terminateInstance()
