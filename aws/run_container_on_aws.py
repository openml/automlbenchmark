#!/bin/python3

import boto3

key = "laptop"
sec = "launch-wizard-7"
instance = "t2.micro"
image = "ami-58d7e821"
dockerImage = "jnkthms/rf"
openmlid = 1
runtime = 1
cores = 1
apikey = "..."


setup = '#!/bin/bash\napt-get update\napt-get install apt-transport-https ca-certificates curl software-properties-common\ncurl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -\nadd-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"\napt-get update\napt-get install -y docker-ce\nusermod -aG docker $USER\ndocker run --rm'
setup = " ".join([setup, dockerImage, str(openmlid), str(runtime), str(cores), apikey, ">> file.csv"])
ec2 = boto3.resource("ec2")

inst = ec2.create_instances(
  ImageId = image,
  MinCount = 1,
  MaxCount = 1,
  InstanceType = instance,
  KeyName = key,
  SecurityGroupIds = [sec],
  UserData = setup
  )
