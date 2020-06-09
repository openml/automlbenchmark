#!/usr/bin/env bash
HERE=$(dirname "$0")
AMLB_DIR="$1"
VERSION=${2:-"latest"}

. $HERE/../shared/setup.sh $HERE

PIP install --no-cache-dir -r $HERE/requirements.txt
if [[ "$VERSION" == "latest" ]]; then
    PIP install scikit-learn
else
    PIP install scikit-learn==${VERSION}
fi
