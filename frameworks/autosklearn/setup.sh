#!/usr/bin/env bash
HERE=$(dirname "$0")
AMLB_DIR="$1"
VERSION=${2:-"latest"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="master"
fi

# creating local venv
. $HERE/../shared/setup.sh $HERE

if [[ -x "$(command -v apt-get)" ]]; then
    SUDO apt-get install -y build-essential swig
fi

if [[ "$VERSION" =~ ^[0-9] ]]; then
    PIP install --no-cache-dir -U auto-sklearn==${VERSION}
else
    PIP install --no-cache-dir -U -e git+https://github.com/automl/auto-sklearn.git@${VERSION}#egg=auto-sklearn
fi

