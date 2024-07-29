#!/usr/bin/env bash
HERE=$(dirname "$0")

#create venv
. ${HERE}/.setup/setup_env
. ${HERE}/../shared/setup.sh ${HERE} true
PIP install --upgrade pip
PIP install --no-cache-dir -r $HERE/requirements.txt