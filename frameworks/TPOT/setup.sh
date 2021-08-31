#!/usr/bin/env bash
HERE=$(dirname "$0")
VERSION=${1:-"stable"}
REPO=${2:-"https://github.com/EpistasisLab/tpot"}
PKG=${3:-"tpot[dask]"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="master"
fi

# creating local venv
. ${HERE}/../shared/setup.sh ${HERE} true

RAWREPO=$(echo ${REPO} | sed "s/github\.com/raw\.githubusercontent\.com/")
if [[ "$VERSION" == "stable" ]]; then
    PIP install --no-cache-dir -U ${PKG}
    VERSION=$(PIP show tpot | awk '/Version/ {print $2}')
    PIP install --no-cache-dir -U -r "${RAWREPO}/v${VERSION}/optional-requirements.txt"
elif [[ "$VERSION" =~ ^[0-9] ]]; then
    PIP install --no-cache-dir -U -r "${RAWREPO}/v${VERSION}/optional-requirements.txt"
    PIP install --no-cache-dir -U ${PKG}==${VERSION}
else
    PIP install --no-cache-dir -U -r "${RAWREPO}/${VERSION}/optional-requirements.txt"
#    PIP install --no-cache-dir -e git+${REPO}@${VERSION}#egg=${PKG}
    LIB=$(echo ${PKG} | sed "s/\[.*\]//")
    TARGET_DIR="${HERE}/lib/${LIB}"
    rm -Rf ${TARGET_DIR}
    git clone --depth 1 --single-branch --branch ${VERSION} --recurse-submodules ${REPO} ${TARGET_DIR}
    PIP install -U -e ${HERE}/lib/${PKG}
fi

PY -c "from tpot import __version__; print(__version__)" >> "${HERE}/.setup/installed"
