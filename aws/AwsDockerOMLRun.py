#!/bin/python3

import boto3
import re

class AwsDockerOMLRun:

  setup = '#!/bin/bash\napt-get update\napt-get install apt-transport-https ca-certificates curl software-properties-common\ncurl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -\nadd-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"\napt-get update\napt-get install -y docker-ce\nusermod -aG docker $USER\ndocker run --rm'
  instance = None
  token = "THIS_IS_A_DUMMY_TOKEN"

  def __init__(self, ssh_key, sec_group, aws_instance_type, aws_instance_image, docker_image, openml_id, runtime, cores, openml_apikey):
    self.ssh_key = ssh_key
    self.sec_group = sec_group
    self.aws_instance_type = aws_instance_type
    self.aws_instance_image = aws_instance_image
    self.docker_image = docker_image
    self.openml_id = openml_id
    self.runtime = runtime
    self.cores = cores
    self.openml_apikey = openml_apikey
    self.ec2_resource = boto3.resource("ec2") # Maybe this should be a class variable, not sure

  def createInstanceRun(self):
    setup = " ".join([self.setup, self.docker_image, str(self.openml_id), str(self.runtime), str(self.cores), self.openml_apikey])
    if self.instance is not None:
      print("Instance already exists, terminate existing instance")
      self.terminateInstance()

    self.instance = self.ec2_resource.create_instances(
      ImageId = self.aws_instance_image,
      MinCount = 1,
      MaxCount = 1,
      InstanceType = self.aws_instance_type,
      KeyName = self.ssh_key,
      SecurityGroupIds = [self.sec_group],
      UserData = setup)[0]

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
      out = self.instance.console_output()
      if "Output" in out.keys():
        out = out["Output"].splitlines()
        out = [x for x in out if re.search(self.token, x)][0]
        return out.split(" ")[-1]

  def __del__(self):
    self.terminateInstance()

if __name__ == "main":

  from time import sleep
  from os import popen

  key = "laptop" #ssh key
  sec = "launch-wizard-7" # security group
  instance = "t2.micro" # instance type
  image = "ami-58d7e821" # aws instance image
  dockerImage = "jnkthms/rf" # docker image
  openmlid = 59
  runtime = 1
  cores = 1
  apikey = popen("cat ~/.openml/config | grep apikey").read().split("=")[1][:-1] # openml apikey

  run = AwsDockerOMLRun(ssh_key = key, sec_group = sec, aws_instance_type = instance,
    aws_instance_image = image, docker_image = dockerImage, openml_id = openmlid, runtime = runtime, cores = cores,
    openml_apikey = apikey)

  run.createInstanceRun()
  for i in range(100):
    print(i)
    sleep(10)
    run.getResult()
  run.terminateInstance()
