#!/usr/bin/env bash
HERE=$(dirname "$0")
VERSION=${1:-"stable"}
REPO=${2:-"https://github.com/openml-labs/gama"}
PKG=${3:-"gama"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="master"
fi

#create local venv
. ${HERE}/../shared/setup.sh ${HERE} true

PIP install -r ${HERE}/requirements.txt
if [[ "$VERSION" == "stable" ]]; then
    PIP install --no-cache-dir -U ${PKG}
elif [[ "$VERSION" =~ ^[0-9] ]]; then
    PIP install --no-cache-dir -U ${PKG}==${VERSION}
else
#    PIP install --no-cache-dir -e git+${REPO}@${VERSION}#egg=${PKG}
    TARGET_DIR="${HERE}/lib/${PKG}"
    rm -Rf ${TARGET_DIR}
    git clone --depth 1 --single-branch --branch ${VERSION} --recurse-submodules ${REPO} ${TARGET_DIR}
    PIP install -U -e ${TARGET_DIR}
fi

if [[ "$VERSION" == "23.0.0" ]]; then
    # We include this only because this is the fixed version for the 2023Q2 definition.
    echo "GAMA/setup.sh: Downgrading scikit-learn to compatible version."
    PIP install --no-cache-dir -U "scikit-learn<1.3"
fi

PY -c "from gama import __version__; print(__version__)" >> "${HERE}/.setup/installed"
