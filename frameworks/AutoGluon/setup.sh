#!/usr/bin/env bash

HERE=$(dirname "$0")
VERSION=${1:-"stable"}
REPO=${2:-"https://github.com/awslabs/autogluon.git"}
PKG=${3:-"autogluon"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="master"
fi

# creating local venv
. ${HERE}/../shared/setup.sh ${HERE} true

# Below fixes seg fault on MacOS due to bug in libomp: https://github.com/awslabs/autogluon/issues/1442
if [[ -x "$(command -v brew)" ]]; then
    wget https://raw.githubusercontent.com/Homebrew/homebrew-core/fb8323f2b170bd4ae97e1bac9bf3e2983af3fdb0/Formula/libomp.rb -P "${HERE}/lib"
    brew install "${HERE}/lib/libomp.rb"
fi

PIP install --upgrade pip
PIP install --upgrade setuptools wheel

if [[ "$VERSION" == "stable" ]]; then
    PIP install --no-cache-dir -U "${PKG}"
    PIP install --no-cache-dir -U "${PKG}.tabular[skex]"
elif [[ "$VERSION" =~ ^[0-9] ]]; then
    PIP install --no-cache-dir -U "${PKG}==${VERSION}"
    PIP install --no-cache-dir -U "${PKG}.tabular[skex]==${VERSION}"
else
    TARGET_DIR="${HERE}/lib/${PKG}"
    rm -Rf ${TARGET_DIR}
    git clone --depth 1 --single-branch --branch ${VERSION} --recurse-submodules ${REPO} ${TARGET_DIR}
    cd ${TARGET_DIR}
    PY_EXEC_NO_ARGS="$(cut -d' ' -f1 <<<"$py_exec")"
    PY_EXEC_DIR=$(dirname "$PY_EXEC_NO_ARGS")
    env PATH="$PY_EXEC_DIR:$PATH" bash -c ./full_install.sh
    PIP install -e tabular/[skex]
fi

PY -c "from autogluon.tabular.version import __version__; print(__version__)" >> "${HERE}/.setup/installed"
