#!/usr/bin/env bash
HERE=$(dirname "$0")
AMLB_DIR="$1"
VERSION=${2:-"latest"}
REPO=${3:-"https://github.com/PGijsbers/gama"}
PKG=${4:-"gama"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="master"
fi

#create local venv
. $HERE/../shared/setup.sh $HERE
#. $AMLB_DIR/frameworks/shared/setup.sh $HERE

PIP install -r $HERE/requirements.txt
if [[ "$VERSION" =~ ^[0-9] ]]; then
    PIP install --no-cache-dir ${PKG}==${VERSION}
else
#    PIP install --no-cache-dir -e git+${REPO}@${VERSION}#egg=${PKG}
    TARGET_DIR="${HERE}/lib/${PKG}"
    rm -Rf ${TARGET_DIR}
    git clone --depth 1 --single-branch --branch ${VERSION} --recurse-submodules ${REPO} ${TARGET_DIR}
    PIP install -e ${TARGET_DIR}
fi
