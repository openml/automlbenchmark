#!/usr/bin/env bash
pip3 install --no-cache-dir -r requirements.txt
DOWNLOAD_DIR="./libs"
TARGET_DIR="$DOWNLOAD_DIR/hyperopt-sklearn"
if [[ ! -e "$TARGET_DIR" ]]; then
    git clone https://github.com/hyperopt/hyperopt-sklearn.git $TARGET_DIR
fi
pip3 install --no-cache-dir -r automl/frameworks/hyperoptsklearn/py_requirements.txt
pip3 install --no-cache-dir -e $TARGET_DIR
