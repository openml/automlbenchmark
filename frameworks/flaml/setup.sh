#!/usr/bin/env bash
HERE=$(dirname "$0")
VERSION=${1:-"stable"}
REPO=${2:-"https://github.com/microsoft/FLAML.git"}
PKG=${3:-"flaml"}


. ${HERE}/../shared/setup.sh ${HERE} true

if [[ "$VERSION" == "latest" ]]; then
    VERSION="main"
    OPTIONALS=""
fi

if [[ "$VERSION" == "benchmark" ]]; then
    VERSION="stable"
    OPTIONALS="[catboost]"
else
    PIP uninstall -y catboost
fi


if [[ "$VERSION" == "stable" ]]; then
    PIP install --no-cache-dir -U "${PKG}${OPTIONALS}<2"
elif [[ "$VERSION" =~ ^[0-9] ]]; then
    PIP install --no-cache-dir -U ${PKG}${OPTIONALS}==${VERSION}
else
#    PIP install --no-cache-dir -e git+${REPO}@${VERSION}#egg=${PKG}
    TARGET_DIR="${HERE}/lib/${PKG}"
    rm -Rf ${TARGET_DIR}
    git clone --depth 1 --single-branch --branch ${VERSION} --recurse-submodules ${REPO} ${TARGET_DIR}
    PIP install -U -e ${TARGET_DIR}${OPTIONALS}
fi

PY -c "from flaml import __version__; print(__version__)" >> "${HERE}/.setup/installed"
echo ${OPTIONALS} >> "${HERE}/.setup/installed"
