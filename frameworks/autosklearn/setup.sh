#!/usr/bin/env bash
HERE=$(dirname "$0")
VERSION=${1:-"stable"}
REPO=${2:-"https://github.com/automl/auto-sklearn.git"}
PKG=${3:-"auto-sklearn"}
if [[ "$VERSION" == "stable" ]]; then
    VERSION="master"
fi

# creating local venv
. ${HERE}/../shared/setup.sh ${HERE} true

if [[ -x "$(command -v apt-get)" ]]; then
    SUDO apt-get install -y build-essential swig
fi

PIP install --no-cache-dir -r $HERE/requirements.txt

if [[ "$VERSION" == "stable" ]]; then
    PIP install --no-cache-dir -U ${PKG}
elif [[ "$VERSION" =~ ^[0-9] ]]; then
    # Provided a version number
    PIP install --no-cache-dir -U ${PKG}==${VERSION}
else
    TARGET_DIR="${HERE}/lib/${PKG}"
    rm -Rf ${TARGET_DIR}
    git clone  --recurse-submodules ${REPO} ${TARGET_DIR}
    cd ${TARGET_DIR}
    git checkout $VERSION
    cd ${HERE}
    PIP install -U -e ${TARGET_DIR}
fi

PY -c "from autosklearn import __version__; print(__version__)" >> "${HERE}/.installed"
