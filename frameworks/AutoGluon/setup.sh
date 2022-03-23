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
    wget https://raw.githubusercontent.com/Homebrew/homebrew-core/fb8323f2b170bd4ae97e1bac9bf3e2983af3fdb0/Formula/libomp.rb
    brew install libomp.rb
    rm libomp.rb
fi

PIP install --upgrade pip
PIP install --upgrade setuptools wheel
PIP install "scikit-learn-intelex<2021.6"

if [[ "$VERSION" == "stable" ]]; then
    PIP install --no-cache-dir -U ${PKG}
elif [[ "$VERSION" =~ ^[0-9] ]]; then
    PIP install --no-cache-dir -U ${PKG}==${VERSION}
else
    TARGET_DIR="${HERE}/lib/${PKG}"
    rm -Rf ${TARGET_DIR}
    git clone --depth 1 --single-branch --branch ${VERSION} --recurse-submodules ${REPO} ${TARGET_DIR}
    cd ${TARGET_DIR}
    # Note: Normally we would just call `./full_install.sh` but because `pip` and `PIP` are not the same,
    # the script does not work here. Therefore, we have to explicitly install each submodule below.
    # This has the downside that source installs of old versions of the package might not follow the below steps.
    # It is recommended to instead use pip install of older AG versions to ensure it works correctly.
    # https://github.com/awslabs/autogluon/blob/master/full_install.sh
    PIP install -e common/
    PIP install -e features/
    PIP install -e core/[all]
    PIP install -e tabular/[all]
    PIP install -e text/
    PIP install -e vision/
    PIP install -e autogluon/
fi

PY -c "from autogluon.tabular.version import __version__; print(__version__)" >> "${HERE}/.setup/installed"
