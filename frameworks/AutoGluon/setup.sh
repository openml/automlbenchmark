#!/usr/bin/env bash

# exit when any command fails
set -e

# Uncomment for debugging installation
# set -x

HERE=$(dirname "$0")
VERSION=${1:-"stable"}
REPO=${2:-"https://github.com/autogluon/autogluon.git"}
PKG=${3:-"autogluon"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="master"
fi

# creating local venv
. ${HERE}/../shared/setup.sh ${HERE} true

# aka "{xyz}/automlbenchmark/frameworks/AutoGluon/venv/bin/python"
PY_EXEC_NO_ARGS="$(cut -d' ' -f1 <<<"$py_exec")"

# Below fixes seg fault on MacOS due to bug in libomp: https://github.com/autogluon/autogluon/issues/1442
if [[ -x "$(command -v brew)" ]]; then
    wget https://raw.githubusercontent.com/Homebrew/homebrew-core/fb8323f2b170bd4ae97e1bac9bf3e2983af3fdb0/Formula/libomp.rb -P "${HERE}/lib"
    brew install "${HERE}/lib/libomp.rb"
fi

PIP install --upgrade pip

PIP install uv
UV="${PY_EXEC_NO_ARGS} -m uv"

if [[ "$VERSION" == "stable" ]]; then
    $UV pip install --no-cache-dir -U "${PKG}"
    $UV pip install --no-cache-dir -U "${PKG}.tabular[skex]"
elif [[ "$VERSION" =~ ^[0-9] ]]; then
    $UV pip install --no-cache-dir -U "${PKG}==${VERSION}"
    $UV pip install --no-cache-dir -U "${PKG}.tabular[skex]==${VERSION}"
else
    TARGET_DIR="${HERE}/lib/${PKG}"
    rm -Rf ${TARGET_DIR}
    git clone --depth 1 --single-branch --branch ${VERSION} --recurse-submodules ${REPO} ${TARGET_DIR}
    cd ${TARGET_DIR}
    PY_EXEC_DIR=$(dirname "$PY_EXEC_NO_ARGS")

    # Install in non-editable mode to avoid interaction with other pre-existing AutoGluon installations
    env PATH="$PY_EXEC_DIR:$PATH" bash -c "./full_install.sh --non-editable"
    $UV pip install tabular/[skex]
fi

# Note: `setuptools` being present in the venv will cause torch==1.4.x to raise an exception for an unknown reason in AMLB.
echo "Finished setup, testing autogluon install..."

PY -c "from autogluon.tabular.version import __version__; print(__version__)" >> "${HERE}/.setup/installed"
