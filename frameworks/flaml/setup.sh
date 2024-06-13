#!/usr/bin/env bash
HERE=$(dirname "$0")
VERSION=${1:-"stable"}
REPO=${2:-"https://github.com/microsoft/FLAML.git"}
PKG=${3:-"flaml"}


. ${HERE}/../shared/setup.sh ${HERE} true

OPTIONALS="[automl]"
if [[ "$VERSION" == "latest" ]]; then
    VERSION="main"
fi

if [[ "$VERSION" == "benchmark" ]]; then
    VERSION="stable"
    OPTIONALS="[automl, catboost]"
else
    PIP uninstall -y catboost
fi

if [[ "$VERSION" == "stable" ]]; then
    PIP install --no-cache-dir -U "${PKG}${OPTIONALS}"
elif [[ "$VERSION" =~ ^[0-1] ]]; then
    PIP install --no-cache-dir -U ${PKG}${OPTIONALS}==${VERSION}
    # FLAML 1.2.4 does not work with newer versions of xgboost
    PIP install "xgboost<2"
elif [[ "$VERSION" =~ ^[2-9] ]]; then
    PIP install --no-cache-dir -U ${PKG}${OPTIONALS}==${VERSION}
else
    TARGET_DIR="${HERE}/lib/${PKG}"
    rm -Rf ${TARGET_DIR}
    git clone --depth 1 --single-branch --branch ${VERSION} --recurse-submodules ${REPO} ${TARGET_DIR}
    PIP install -U -e ${TARGET_DIR}${OPTIONALS}
fi

PY -c "from flaml import __version__; print(__version__)" >> "${HERE}/.setup/installed"
echo ${OPTIONALS} >> "${HERE}/.setup/installed"
