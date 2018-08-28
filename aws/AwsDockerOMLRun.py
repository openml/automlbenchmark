#!/usr/bin/python3

import boto3
import json
import re


class AwsDockerOMLRun:

    def __init__(self, aws_instance_type, docker_image, openml_id, fold,
                 runtime, cores, metric, region_name=None):

        self.aws_instance_type = aws_instance_type
        self.docker_image = docker_image
        self.openml_id = openml_id
        self.fold = fold
        self.runtime = runtime
        self.cores = cores
        self.metric = metric

        # load config file
        with open("config.json") as file:
                config = json.load(file)

        self.setup = config["setup"]
        self.token = config["token"]

        if region_name is None:
            if "region_name" in config.keys() and len(config["region_name"]) > 0:
                self.region_name = config["region_name"]
            else:
                self.region_name = boto3.session.Session().region_name
        self.ec2_resource = boto3.resource("ec2", region_name=region_name)

        with open("resources/ami.json") as file:
                amis = json.load(file)

        self.aws_instance_image = amis[self.region_name]
        self.instance = None

    def createInstanceRun(self):
        setup = "%s %s -f %i -t %i -s %i -p %i -m %s" % (self.setup,
                                                         self.docker_image,
                                                         self.fold,
                                                         self.openml_id,
                                                         self.runtime,
                                                         self.cores,
                                                         self.metric)
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
            raw_log = self.instance.console_output(Latest=True)
            if "Output" in raw_log.keys():
                out = raw_log["Output"].splitlines()
                out = [x for x in out if re.search(self.token, x)]
                if len(out) == 1:
                    return {"res": out[0].split(" ")[-1], "log": raw_log}


if __name__ == "main":

    from time import sleep

    instance = "m5.xlarge"  # instance type
    dockerImage = "jnkthms/tpot"  # docker image
    openmlid = 59
    runtime = 600
    cores = 4
    run = AwsDockerOMLRun(aws_instance_type=instance,
                          docker_image=dockerImage,
                          openml_id=openmlid,
                          fold=1,
                          runtime=runtime,
                          cores=cores,
                          metric="acc")

    run.createInstanceRun()
    res = []
    for i in range(100):
        print(i)
        sleep(10)
        run.getResult()

    r = run.instance.console_output()

    run.terminateInstance()
