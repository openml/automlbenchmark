#!/usr/bin/env bash
. ../setup/aliases.sh
PIP install --no-cache-dir -r requirements.txt
PIP install --no-cache-dir -r automl/frameworks/TPOT/py_requirements.txt
