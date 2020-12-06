#!/usr/bin/env bash
HERE=$(dirname "$0")
AMLB_DIR="$1"
VERSION=${2:-"latest"}
REPO=${3:-"https://github.com/sberbank-ai-lab/LightAutoML.git"}
PKG=${4:-"lightautoml"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="master"
fi
echo "setting up LightAutoML version $VERSION"
# creating local venv
. ${HERE}/../shared/setup.sh ${HERE}


#cat ${HERE}/requirements.txt | sed '/^$/d' | while read -r i; do PIP install "$i"; done

if [[ "$VERSION" =~ ^[0-9] ]]; then
    PIP install --no-cache-dir ${PKG}==${VERSION}
else
    TARGET_DIR="${HERE}/lib/${PKG}"
    rm -Rf ${TARGET_DIR}
    git clone --depth 1 --single-branch --branch ${VERSION} --recurse-submodules ${REPO} ${TARGET_DIR}
    PIP install -e ${TARGET_DIR}
fi
