#!/usr/bin/env bash
HERE=$(dirname "$0")
VERSION=${1:-"stable"}
REPO=${2:-"https://github.com/sapientml/sapientml"}
PKG=${3:-"sapientml"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="main"
fi

#create local venv
. ${HERE}/../shared/setup.sh ${HERE} true

# PIP install -r ${HERE}/requirements.txt
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

PY -c "import pkg_resources; print(pkg_resources.get_distribution('sapientml').version)" >> "${HERE}/.setup/installed"
