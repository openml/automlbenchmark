#!/bin/bash

echo "*** start ami configuration ***"

apt-get update

echo "*** install: curl, wget, unzip, git ***"
# Skip restart prompt of 'libssl1.1' by running following command
echo '* libraries/restart-without-asking boolean true' | debconf-set-selections
apt-get -y install curl wget unzip git
apt-get -y install software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update

echo "*** install python${PYV} ***"
apt-get -y install python$PYV python$PYV-venv python$PYV-dev python3-pip

echo "*** install awscli ***"
pip3 install -U wheel awscli --no-cache-dir

echo "make automl directory structure"
mkdir -p /s3bucket/input
mkdir -p /s3bucket/output
mkdir -p /s3bucket/user
mkdir /repo

echo "clone repo"
cd /repo
git clone --depth 1 --single-branch --branch $BRANCH $GITREPO .

echo "create python environment"
python3 -m venv venv

echo "install python packages"
/repo/venv/bin/pip3 install -U pip
xargs -L 1 /repo/venv/bin/pip3 install --no-cache-dir < requirements.txt
