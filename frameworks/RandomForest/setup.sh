#!/usr/bin/env bash
HERE=$(dirname "$0")
AMLB_DIR="$1"
VERSION=${2:-"latest"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="master"
fi

# by passing the module directory to `setup.sh`, it tells it to automatically create a virtual env under the current module.
# this virtual env is then used to run the exec.py only, and can be configured here using `PIP` and `PY` commands.
. $HERE/../shared/setup.sh $HERE
#. $AMLB_DIR/frameworks/shared/setup.sh $HERE

#PIP install -r $HERE/requirements.txt
if [[ "$VERSION" =~ ^[0-9] ]]; then
    PIP install --no-cache-dir -U scikit-learn==${VERSION}
else
    PIP install --no-cache-dir -U -e git+https://github.com/scikit-learn/scikit-learn.git@${VERSION}#egg=scikit-learn
fi
