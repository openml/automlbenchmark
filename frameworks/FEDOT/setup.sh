#!/usr/bin/env bash
HERE=$(dirname "$0")
VERSION=${1:-"stable"}
REPO=${2:-"https://github.com/aimclub/FEDOT.git"}
PKG=${3:-"fedot"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="master"
fi

# creating local venv
. ${HERE}/../shared/setup.sh ${HERE} true

RAWREPO=$(echo ${REPO} | sed "s/github\.com/raw\.githubusercontent\.com/")
if [[ "$VERSION" == "stable" ]]; then
    PIP install --no-cache-dir -U ${PKG}
    echo GET_VERSION_STABLE
    VERSION=$(PY -c "${GET_VERSION_STABLE}")
elif [[ "$VERSION" =~ ^[0-9] ]]; then
    PIP install --no-cache-dir -U ${PKG}==${VERSION}
else
    TARGET_DIR="${HERE}/lib/${PKG}"
    rm -Rf ${TARGET_DIR}

    if [[ "$VERSION" =~ ^# ]]; then
      COMMIT="${VERSION:1}"
    else
      # find the latest commit to the VERSION branch
      COMMIT=$(git ls-remote "${REPO}" | grep "refs/heads/${VERSION}" | cut -f 1)
      DEPTH="--depth 1 --branch ${VERSION}"
    fi

    git clone  --recurse-submodules --shallow-submodules ${DEPTH} ${REPO} ${TARGET_DIR}
    cd ${TARGET_DIR}
    git checkout "${COMMIT}"
    git submodule update --init --recursive
    cd ${HERE}
    PIP install -U -e ${TARGET_DIR}
fi

installed="${HERE}/.setup/installed"
PY -c "from fedot import __version__; print(__version__)" >> "$installed"
if [[ -n $COMMIT ]]; then
  truncate -s-1 "$installed"
  echo "#${COMMIT}" >> "$installed"
fi
