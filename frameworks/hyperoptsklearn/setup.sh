#!/usr/bin/env bash
HERE=$(dirname "$0")
VERSION=${1:-"latest"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="master"
fi

. ${HERE}/../shared/setup.sh ${HERE}

PIP install --no-cache-dir -r ${HERE}/requirements.txt
PIP install --no-cache-dir -e git+https://github.com/hyperopt/hyperopt-sklearn.git@${VERSION}#egg=hyperopt-sklearn
