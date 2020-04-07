#!/usr/bin/env bash
VERSION=${1:-"v.0.6.0"}

HERE=$(dirname "$0")
. $HERE/../shared/setup.sh $HERE

if [[ -x "$(command -v apt-get)" ]]; then
    SUDO apt-get install -y build-essential swig
fi
# by passing the module directory to `setup.sh`, it tells it to automatically create a virtual env under the current module.
# this virtual env is then used to run the exec.py only, and can be configured here using `PIP` and `PY` commands.
#curl "https://raw.githubusercontent.com/automl/auto-sklearn/${VERSION}/requirements.txt" | sed '/^$/d' | while read -r i; do PIP install "$i"; done
PIP install --no-cache-dir -r "https://raw.githubusercontent.com/automl/auto-sklearn/${VERSION}/requirements.txt"
PIP install --no-cache-dir -r $HERE/requirements.txt
