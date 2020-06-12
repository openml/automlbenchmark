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
if [[ "$VERSION" =~ ^[0-9] ]]; then
    PIP install --no-cache-dir -U gama==${VERSION}
else
    PIP install --no-cache-dir -U -e git+https://github.com/PGijsbers/gama@${VERSION}#egg=gama
fi
