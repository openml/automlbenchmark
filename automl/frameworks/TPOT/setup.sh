#!/usr/bin/env bash
. $(dirname "$0")/../../setup/shared.sh
PIP install --no-cache-dir -r automl/frameworks/TPOT/py_requirements.txt
