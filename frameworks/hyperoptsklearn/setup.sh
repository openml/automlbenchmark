#!/usr/bin/env bash
HERE=$(dirname "$0")
. $HERE/../shared/setup.sh
TARGET_DIR="$HERE/libs/hyperopt-sklearn"
if [[ ! -e "$TARGET_DIR" ]]; then
    git clone https://github.com/hyperopt/hyperopt-sklearn.git $TARGET_DIR
fi
PIP install --no-cache-dir -e $TARGET_DIR
