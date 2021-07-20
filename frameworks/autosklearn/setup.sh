#!/usr/bin/env bash
HERE=$(dirname "$0")
VERSION=${1:-"stable"}
REPO=${2:-"https://github.com/automl/auto-sklearn.git"}
PKG=${3:-"auto-sklearn"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="development"
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

    if [[ "$VERSION" =~ ^# ]]; then
      COMMIT="${VERSION:1}"
    else
      # find the latest commit to the VERSION branch
      COMMIT=$(git ls-remote "${REPO}" | grep "refs/heads/${VERSION}" | cut -f 1)
    fi

    git clone  --recurse-submodules ${REPO} ${TARGET_DIR}
    cd ${TARGET_DIR}
    git checkout "${COMMIT}"
    cd ${HERE}
    PIP install -U -e ${TARGET_DIR}
fi

PY -c "from autosklearn import __version__; print(__version__)" >> "${HERE}/.installed"
if [[ -n $COMMIT ]]; then
  truncate -s-1 "${HERE}/.installed"
  echo "#${COMMIT}" >> "${HERE}/.installed"
fi
