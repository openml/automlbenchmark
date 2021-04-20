#!/usr/bin/env bash
HERE=$(dirname "$0")
VERSION=${1:-"stable"}
. ${HERE}/../shared/setup.sh "${HERE}"
if [[ -x "$(command -v apt-get)" ]]; then
    SUDO apt-get update
    SUDO apt-get install -y wget unzip openjdk-8-jdk
fi

AUTOWEKA_ARCHIVE="autoweka-${VERSION}.zip"
DOWNLOAD_DIR="$HERE/lib"
TARGET_DIR="$DOWNLOAD_DIR/autoweka"
if [[ ! -e "$TARGET_DIR" ]]; then
    wget http://www.cs.ubc.ca/labs/beta/Projects/autoweka/$AUTOWEKA_ARCHIVE -P $DOWNLOAD_DIR
    unzip $DOWNLOAD_DIR/$AUTOWEKA_ARCHIVE -d $TARGET_DIR
fi

find $HERE/lib/autoweka*.zip | sed -e 's/.*\/autoweka-\(.*\)\.zip/\1/' >> "${HERE}/.installed"
