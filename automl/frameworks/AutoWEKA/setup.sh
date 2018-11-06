#!/usr/bin/env bash
pip3 install --no-cache-dir -r requirements.txt

AUTOWEKA_ARCHIVE="autoweka-2.6.zip"
DOWNLOAD_DIR="./libs"
TARGET_DIR="./libs/autoweka"
if [ ! -e "$TARGET_DIR" ]; then
    wget http://www.cs.ubc.ca/labs/beta/Projects/autoweka/$AUTOWEKA_ARCHIVE -P $DOWNLOAD_DIR
    unzip $DOWNLOAD_DIR/$AUTOWEKA_ARCHIVE -d TARGET_DIR
fi
