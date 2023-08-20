#!/usr/bin/env bash
HERE=$(dirname "$0")
VERSION=${1:-"stable"}
REPO=${2:-"https://github.com/udellgroup/oboe.git"}
PKG=${3:-"oboe"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="master"
fi

. ${HERE}/../shared/setup.sh ${HERE} true

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

#cat ${HERE}/requirements.txt | sed '/^$/d' | while read -r i; do PIP install --no-cache-dir -U "$i"; done
#PIP install --no-cache-dir -U -e git+https://github.com/udellgroup/oboe.git@${VERSION}#egg=oboe
