#!/usr/bin/env bash
shopt -s expand_aliases
HERE=$(dirname "$0")
ROOT_DIR="$1"
. $ROOT_DIR/frameworks/shared/setup.sh
RF_VENV=$HERE/venv
alias PY_RF="$RF_VENV/bin/python3 -W ignore"
if [[ ! -f "$RF_VENV/bin/python3" ]]; then
    python3 -m venv $RF_VENV
    PY_RF -m pip install -U pip
fi
echo "$(command -v PY_RF)"
PY_RF -m pip install --no-cache-dir -r requirements.txt
PY_RF -m pip install --no-cache-dir -U -r $HERE/requirements.txt

