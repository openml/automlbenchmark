#!/usr/bin/env bash
HERE=$(dirname "$0")
VERSION=${1:-"stable"}
REPO=${2:-"https://github.com/hyperopt/hyperopt-sklearn.git"}
PKG=${3:-"hyperopt-sklearn"}

if [[ "$VERSION" == "latest" || "$VERSION" == "stable" ]]; then
    VERSION="master"
fi

. ${HERE}/../shared/setup.sh ${HERE} true

if [[ "$VERSION" =~ ^[0-9] ]]; then
    PIP install git+https://github.com/hyperopt/hyperopt-sklearn@${VERSION}
else
#    PIP install --no-cache-dir -e git+${REPO}@${VERSION}#egg=${PKG}
    LIB=$(echo ${PKG} | sed "s/\[.*\]//")
    TARGET_DIR="${HERE}/lib/${LIB}"
    rm -Rf ${TARGET_DIR}
    if [[ "$VERSION" =~ ^# ]]; then
        git clone --recurse-submodules --shallow-submodules ${REPO} ${TARGET_DIR}
        cd ${TARGET_DIR}
        git checkout "${VERSION:1}"
        cd ${HERE}
    else
        git clone --depth 1 --single-branch --branch ${VERSION} --recurse-submodules ${REPO} ${TARGET_DIR}
    fi
    PIP install -U -e ${HERE}/lib/${PKG}
fi
