#!/usr/bin/env bash
HERE=$(dirname "$0")
VERSION=${1:-"latest"}
REPO=${2:-"https://github.com/sberbank-ai-lab/LightAutoML.git"}
PKG=${3:-"lightautoml"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="master"
fi

# creating local venv
. ${HERE}/../shared/setup.sh ${HERE} true

apt-get install -y python3-opencv

if [[ "$VERSION" == "stable" ]]; then
    PIP install --no-cache-dir -U ${PKG}
elif [[ "$VERSION" =~ ^[0-9] ]]; then
    PIP install --no-cache-dir -U ${PKG}==${VERSION}
else
    TARGET_DIR="${HERE}/lib/${PKG}"
    rm -Rf ${TARGET_DIR}
    git clone --depth 1 --single-branch --branch ${VERSION} --recurse-submodules ${REPO} ${TARGET_DIR}
    cd "${TARGET_DIR}"
    bash build_package.sh
fi

PY -c "import matplotlib; matplotlib.use('agg'); from lightautoml import __version__; print(__version__)" >> "${HERE}/.setup/installed"
