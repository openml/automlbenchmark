#!/usr/bin/env bash
HERE=$(dirname "$0")
AMLB_DIR="$1"
VERSION=${2:-"latest"}
REPO=${3:-"https://github.com/EpistasisLab/tpot"}
PKG=${4:-"tpot[dask]"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="master"
fi

# creating local venv
. $HERE/../shared/setup.sh $HERE

RAWREPO=$(echo ${REPO} | sed "s/github\.com/raw\.githubusercontent\.com/")
if [[ "$VERSION" =~ ^[0-9] ]]; then
    PIP install --no-cache-dir -U -r "${RAWREPO}/v${VERSION}/optional-requirements.txt"
    PIP install --no-cache-dir ${PKG}==${VERSION}
else
    PIP install --no-cache-dir -U -r "${RAWREPO}/${VERSION}/optional-requirements.txt"
#    PIP install --no-cache-dir -e git+${REPO}@${VERSION}#egg=${PKG}
    LIB=$(echo ${PKG} | sed "s/\[.*\]//")
    TARGET_DIR="${HERE}/lib/${LIB}"
    rm -Rf ${TARGET_DIR}
    git clone --depth 1 --single-branch --branch ${VERSION} --recurse-submodules ${REPO} ${TARGET_DIR}
    PIP install -e ${HERE}/lib/${PKG}
fi
