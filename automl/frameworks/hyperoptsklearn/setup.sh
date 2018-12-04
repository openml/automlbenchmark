#!/usr/bin/env bash
if ! [[ -x "$(command -v PIP)" ]]; then
    alias PIP=pip3
fi
PIP install --no-cache-dir -r requirements.txt
DOWNLOAD_DIR="./libs"
TARGET_DIR="$DOWNLOAD_DIR/hyperopt-sklearn"
if [[ ! -e "$TARGET_DIR" ]]; then
    git clone https://github.com/hyperopt/hyperopt-sklearn.git $TARGET_DIR
fi
PIP install --no-cache-dir -r automl/frameworks/hyperoptsklearn/py_requirements.txt
PIP install --no-cache-dir -e $TARGET_DIR
