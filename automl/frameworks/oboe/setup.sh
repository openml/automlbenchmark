#!/usr/bin/env bash
if ! [[ -x "$(command -v PIP)" ]]; then
    alias PIP=pip3
fi
PIP install --no-cache-dir -r requirements.txt
DOWNLOAD_DIR="./libs"
TARGET_DIR="$DOWNLOAD_DIR/oboe"
if [[ ! -e "$TARGET_DIR" ]]; then
    git clone https://github.com/udellgroup/oboe.git $TARGET_DIR
fi
PIP install --no-cache-dir -e $TARGET_DIR
PIP install --no-cache-dir -r automl/frameworks/oboe/py_requirements.txt