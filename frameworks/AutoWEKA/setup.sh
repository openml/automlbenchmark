#!/usr/bin/env bash
HERE=$(dirname "$0")
VERSION=${1:-"stable"}
. ${HERE}/../shared/setup.sh "${HERE}"
if [[ -x "$(command -v apt-get)" ]]; then
    SUDO apt-get update
    SUDO apt-get install -y wget unzip openjdk-8-jdk
fi

if [[ "$VERSION" == "latest" || "$VERSION" == "stable" ]]; then
    URL="https://raw.githubusercontent.com/automl/autoweka/master"
    AUTOWEKA_ARCHIVE="autoweka-latest.zip"
else
    URL="http://www.cs.ubc.ca/labs/beta/Projects/autoweka"
    AUTOWEKA_ARCHIVE="autoweka-${VERSION}.zip"
fi

DOWNLOAD_DIR="$HERE/lib"
TARGET_DIR="$DOWNLOAD_DIR/autoweka"
rm -Rf ${TARGET_DIR}

wget "$URL/$AUTOWEKA_ARCHIVE" -P $DOWNLOAD_DIR
unzip "$DOWNLOAD_DIR/$AUTOWEKA_ARCHIVE" -d $TARGET_DIR

find $HERE/lib/autoweka*.zip | sed -e 's/.*\/autoweka-\(.*\)\.zip/\1/' >> "${HERE}/.installed"
