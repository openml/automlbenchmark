#!/usr/bin/env bash
HERE=$(dirname "$0")
. $HERE/../shared/setup.sh
if [[ -x "$(command -v apt-get)" ]]; then
    SUDO apt-get install -y build-essential swig
fi
curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | sed '/^$/d' | while read -r i; do PIP install "$i"; done
PIP install --no-cache-dir -r $HERE/requirements.txt

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
