#!/usr/bin/env bash
HERE=$(dirname "$0")
. $HERE/../shared/setup.sh
PIP install --no-cache-dir -r $HERE/requirements.txt

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
