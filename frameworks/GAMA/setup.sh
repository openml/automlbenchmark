#!/usr/bin/env bash
HERE=$(dirname "$0")
AMLB_DIR="$1"
VERSION=${2:-"latest"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="master"
fi

#create local venv
. $HERE/../shared/setup.sh $HERE
#. $AMLB_DIR/frameworks/shared/setup.sh $HERE

PIP install -r $HERE/requirements.txt
if [[ "$VERSION" == "latest" ]]; then
    PIP install --no-cache-dir -U -e git+https://github.com/PGijsbers/gama@i${VERSION}#egg=gama
else
    PIP install --no-cache-dir -U gama==${VERSION}
fi
