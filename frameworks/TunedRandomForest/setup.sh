#!/usr/bin/env bash
HERE=$(dirname "$0")
AMLB_DIR="$1"
VERSION=${2:-"latest"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="master"
fi

. $HERE/../shared/setup.sh $HERE

PIP install --no-cache-dir -r $HERE/requirements.txt
if [[ "$VERSION" =~ ^[0-9] ]]; then
    PIP install --no-cache-dir -U scikit-learn==${VERSION}
else
    PIP install --no-cache-dir -U -e git+https://github.com/scikit-learn/scikit-learn.git@${VERSION}#egg=scikit-learn
fi
