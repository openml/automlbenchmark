#!/usr/bin/env bash
HERE=$(dirname "$0")
. $HERE/../shared/setup.sh
RF_VENV=~/venvs/bench/rf
alias PY_RF="$RF_VENV/bin/python3 -W ignore"
if [[ ! -f "$RF_VENV/bin/python3" ]]; then
    python3 -m venv $RF_VENV
    PY_RF -m pip install -U pip
fi
PY_RF -m pip install --no-cache-dir -r requirements.txt
PY_RF -m pip install --no-cache-dir -U -r $HERE/requirements.txt

