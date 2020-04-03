#!/usr/bin/env bash
HERE=$(dirname "$0")
#AMLB_DIR="$1"
. $HERE/../shared/setup.sh
if [[ -x "$(command -v apt-get)" ]]; then
    SUDO apt-get install -y build-essential swig
fi
# by passing the module directory to `setup.sh`, it tells it to automatically create a virtual env under the current module.
# this virtual env is then used to run the exec.py only, and can be configured here using `PIP` and `PY` commands.
. $HERE/../shared/setup.sh $HERE
PIP install --no-cache-dir -r $HERE/requirements.txt
