#!/usr/bin/env bash
if ! [[ -x "$(command -v PIP)" ]]; then
    alias PIP=pip3
fi
PIP install --no-cache-dir -r requirements.txt
PIP install --no-cache-dir -r automl/frameworks/TPOT/py_requirements.txt
