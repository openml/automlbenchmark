#!/usr/bin/env bash
HERE=$(dirname "$0")
VERSION=${1:-"stable"}
REPO=${2:-"https://github.com/microsoft/FLAML.git"}
PKG=${3:-"flaml"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="main"
fi

. ${HERE}/../shared/setup.sh ${HERE} true

if [[ "$VERSION" == "stable" ]]; then
    PIP install --no-cache-dir -U ${PKG}[benchmark]
elif [[ "$VERSION" =~ ^[0-9] ]]; then
    PIP install --no-cache-dir -U ${PKG}==${VERSION}[benchmark]
else
#    PIP install --no-cache-dir -e git+${REPO}@${VERSION}#egg=${PKG}
    TARGET_DIR="${HERE}/lib/${PKG}"
    rm -Rf ${TARGET_DIR}
    git clone --depth 1 --single-branch --branch ${VERSION} --recurse-submodules ${REPO} ${TARGET_DIR}
    PIP install -U -e ${TARGET_DIR}[benchmark]
fi

PY -c "from flaml import __version__; print(__version__)" >> "${HERE}/.setup/installed"
