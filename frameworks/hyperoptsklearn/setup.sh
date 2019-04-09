#!/usr/bin/env bash
HERE=$(dirname "$0")
. $HERE/../shared/setup.sh
TARGET_DIR="$HERE/lib/hyperopt-sklearn"
if [[ ! -e "$TARGET_DIR" ]]; then
    git clone https://github.com/hyperopt/hyperopt-sklearn.git $TARGET_DIR
fi
PIP install --no-cache-dir -e $TARGET_DIR
PIP install --no-cache-dir -r $HERE/requirements.txt

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
