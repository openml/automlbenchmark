#!/usr/bin/env bash
HERE=$(dirname "$0")
VERSION=${1:-"stable"}
REPO=${2:-"https://github.com/scikit-learn/scikit-learn.git"}
PKG=${3:-"scikit-learn"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="main"
fi

# create venv
. ${HERE}/../shared/setup.sh "${HERE}" true

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

PIP install --no-cache-dir -r ${HERE}/requirements.txt
PY -c "from sklearn import __version__; print(__version__)" >> "${HERE}/.setup/installed"
