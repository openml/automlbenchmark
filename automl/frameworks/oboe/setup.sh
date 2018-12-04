#!/usr/bin/env bash
pip3 install --no-cache-dir -r requirements.txt
DOWNLOAD_DIR="./libs"
TARGET_DIR="$DOWNLOAD_DIR/oboe"
if [[ ! -e "$TARGET_DIR" ]]; then
    git clone https://github.com/udellgroup/oboe.git $TARGET_DIR
fi
pip3 install --no-cache-dir -e $TARGET_DIR
pip3 install --no-cache-dir -r automl/frameworks/oboe/py_requirements.txt