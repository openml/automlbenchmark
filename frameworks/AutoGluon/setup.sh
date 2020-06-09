#!/usr/bin/env bash
HERE=$(dirname "$0")
AMLB_DIR="$1"
VERSION=${2:-"latest"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="master"
fi

# creating local venv
. ${HERE}/../shared/setup.sh ${HERE}
#if [[ -x "$(command -v apt-get)" ]]; then
#    SUDO apt-get install -y libomp-dev
if [[ -x "$(command -v brew)" ]]; then
    brew install libomp
fi

cat ${HERE}/requirements.txt | sed '/^$/d' | while read -r i; do PIP install "$i"; done

if [[ "$VERSION" =~ ^[0-9] ]]; then
    PIP install --no-cache-dir -U autogluon==${VERSION}
else
    PIP install --no-cache-dir -U -e git+https://github.com/awslabs/autogluon.git@${VERSION}#egg=autogluon
fi
