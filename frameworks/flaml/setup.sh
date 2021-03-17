#!/usr/bin/env bash
HERE=$(dirname "$0")
#AMLB_DIR="$1"
# by passing the module directory to `setup.sh`, it tells it to automatically create a virtual env under the current module.
# this virtual env is then used to run the exec.py only, and can be configured here using `PIP` and `PY` commands.
. $HERE/../shared/setup.sh $HERE
#. $AMLB_DIR/frameworks/shared/setup.sh $HERE
PIP install  flaml==0.2.3