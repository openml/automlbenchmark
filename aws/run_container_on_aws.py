#!/bin/python3

import boto3
from time import sleep
from os import popen

key = "laptop" #ssh key
sec = "launch-wizard-7" # security group
instance = "t2.micro" # instance type
image = "ami-58d7e821" # aws instance image
dockerImage = "jnkthms/rf" # docker image
openmlid = 1
runtime = 1
cores = 1
apikey = popen("cat ~/.openml/config | grep apikey").read().split("=")[1][:-2] # openml apikey


setup = '#!/bin/bash\napt-get update\napt-get install apt-transport-https ca-certificates curl software-properties-common\ncurl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -\nadd-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"\napt-get update\napt-get install -y docker-ce\nusermod -aG docker $USER\ndocker run --rm'
setup = " ".join([setup, dockerImage, str(openmlid), str(runtime), str(cores), apikey, ">> /home/ubuntu/file.csv"])
ec2 = boto3.resource("ec2")

inst = ec2.create_instances(
  ImageId = image,
  MinCount = 1,
  MaxCount = 1,
  InstanceType = instance,
  KeyName = key,
  SecurityGroupIds = [sec],
  UserData = setup
)[0]

sleep(300 + runtime) # setup time + runtime
inst.load() # update instance state in boto3

res = popen('scp "%s:%s" "%s"' % ("ubuntu@" + inst.public_ip_address, "~/file.csv", "/dev/stdout")).read()

inst.terminate()
