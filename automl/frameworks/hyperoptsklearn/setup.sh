#!/usr/bin/env bash
. $(dirname "$0")/../../setup/shared.sh
DOWNLOAD_DIR="./libs"
TARGET_DIR="$DOWNLOAD_DIR/hyperopt-sklearn"
if [[ ! -e "$TARGET_DIR" ]]; then
    git clone https://github.com/hyperopt/hyperopt-sklearn.git $TARGET_DIR
fi
PIP install --no-cache-dir -e $TARGET_DIR
