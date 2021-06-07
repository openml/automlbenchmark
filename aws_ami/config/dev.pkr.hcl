# AMI that adds additional packages to run R and Java based frameworks, extends the base AMI.
variable "source_ami" {
  type    = string
  description = "Ubuntu Server 18.04 LTS (HVM), EBS General Purpose (SSD) VolumeType"
  default = "ami-0aa7ef746644a86b6"

  # Optional: define a filter to automatically pick the latest version of an ami as source ami.
  // source_ami_filter = {
  //   filters = {
  //     name                = "automl-base-*"
  //     root-device-type    = "ebs"
  //     virtualization-type = "hvm"
  //   }
  //   most_recent = true
  //   // owners      = ["099720109477"]
  // }
}

locals { timestamp = regex_replace(timestamp(), "[- TZ:]", "") }

# source blocks configure your builder plugins; your source is then used inside
# build blocks to create resources. A build block runs provisioners and
# post-processors on an instance created by the source.
source "amazon-ebs" "automl-ami" {
  # the profile to use in the shared credentials file for AWS.
  // profile       = "default"

  ami_name      = "automl-dev-${local.timestamp}"
  ami_description = "AMI for the AutoML benchmark project"

  # uncomment following line to create a public ami, default a private ami is created
  // ami_groups = ["all"]
  
  instance_type = "t2.micro"
  source_ami    = var.source_ami

  ssh_username = "ubuntu"
}

# a build block invokes sources and runs provisioning steps on them.
build {
  sources = ["source.amazon-ebs.automl-ami"]

  provisioner "shell" {
    execute_command = "echo 'packer' | sudo -S env {{ .Vars }} {{ .Path }}"
    environment_vars = [
        "BRANCH=stable",
        "GITREPO=https://github.com/openml/automlbenchmark",
        "PYV=3"
    ]
    script = "./scripts/configure-dev.sh"
  }
}