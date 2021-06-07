# Auto-ML AWS AMI

This directory contains the instructions and configuration files to build a custom automl AWS Amazon machine images (AMIs).

There are two AMIs to available:
 - `automl-base`: an AMI with the bare minimum required dependencies installed to run the benchmark.
 - `automl-dev`: an AMI which installs some commonly used software on top. Currently this is just the `R` environment used by all `R` frameworks.

The `automl-dev` AMI builds on the most recent `automl-base` AMI, so make sure `automl-base` is up-to-date before building `automl-dev`.

**Note:** The current configuration only works in the eu-central-1 region of AWS as the base ami is hard coded to match an AMI that's only available in this region.


## Prerequisites

### Install Packer

[Packer](https://learn.hashicorp.com/packer) is the command line tool that's used by this module to build a custom AWS AMI. 
To build the image, the packer cmd tool must be installed on your local machine.
For more install information, [how to install packer](https://learn.hashicorp.com/tutorials/packer/getting-started-install)

### AWS credentials setup
Before building a new automl AMI, the AWS credentials must be configured to allow for programmatic access
- configure AWS credentials
    - create AWS profile (e.g. automl)
- set profile name (2 options)
    - update profile name in packer config file (i.e. base.pkr.hcl)
    - set profile environment variable (e.g. 'export AWS_PROFILE=automl')


## Validate and Build
Before building the automl AMI, the packer config file should be validated.
To validate, run the following command.

```sh
    packer validate ./config/base.pkr.hcl
```

If the validation step has succeeded, run the following command to build the ami.

```sh
    packer build ./config/base.pkr.hcl
```

*note:* Examples are written for `base.pkr.hcl` but also apply to `dev.pkr.hcl`.
