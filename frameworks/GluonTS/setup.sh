#!/usr/bin/env bash

HERE=$(dirname "$0")
VERSION=${1:-"stable"}
REPO=${2:-"https://github.com/awslabs/gluonts.git"}
PKG=${3:-"gluonts"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="dev"
fi

# creating local venv
. ${HERE}/../shared/setup.sh ${HERE} true

PIP install --upgrade pip
PIP install --upgrade setuptools wheel

# TOPIC: installing Prophet
echo "installing Prophet..."
PIP install "prophet"
# -------------------------

if [[ "$VERSION" == "stable" ]]; then
    PIP install --no-cache-dir -U "${PKG}[mxnet,pro,Prophet]"
elif [[ "$VERSION" =~ ^[0-9] ]]; then
    PIP install --no-cache-dir -U "${PKG}[mxnet,pro,Prophet]==${VERSION}"
else
    TARGET_DIR="${HERE}/lib/${PKG}"
    rm -Rf ${TARGET_DIR}
    git clone --depth 1 --single-branch --branch ${VERSION} --recurse-submodules ${REPO} ${TARGET_DIR}
    cd ${TARGET_DIR}
    PY_EXEC_NO_ARGS="$(cut -d' ' -f1 <<<"$py_exec")"
    PY_EXEC_DIR=$(dirname "$PY_EXEC_NO_ARGS")
    env PATH="$PY_EXEC_DIR:$PATH"
    PIP install -e ".[mxnet,pro,Prophet]"
fi


PY -c "from gluonts import __version__; print(__version__)" >> "${HERE}/.setup/installed"
PIP install "mxnet<2.0"
