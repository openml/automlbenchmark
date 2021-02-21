#!/usr/bin/env bash
HERE=$(dirname "$0")
AMLB_DIR="$1"
VERSION=${2:-"stable"}
REPO=${3:-"https://github.com/automl/auto-sklearn.git"}
PKG=${4:-"auto-sklearn"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="master"
fi

# creating local venv
. ${HERE}/../shared/setup.sh ${HERE}

if [[ -x "$(command -v apt-get)" ]]; then
    SUDO apt-get install -y build-essential swig
fi

PIP install --no-cache-dir -r $HERE/requirements.txt

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
# sudo apt-get install python3-dev