#!/usr/bin/env bash
HERE=$(dirname "$0")
VERSION=${2:-"latest"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="master"
fi

. ${HERE}/../shared/setup.sh ${HERE}

TARGET_DIR="$HERE/lib/oboe"
if [[ -e "$TARGET_DIR" ]]; then
    rm -Rf ${TARGET_DIR}
else
    git clone --depth 1 --single-branch --branch ${VERSION} https://github.com/udellgroup/oboe.git ${TARGET_DIR}
fi

cat ${HERE}/requirements.txt | sed '/^$/d' | while read -r i; do PIP install --no-cache-dir "$i"; done
#PIP install --no-cache-dir -e git+https://github.com/udellgroup/oboe.git@${VERSION}#egg=oboe
