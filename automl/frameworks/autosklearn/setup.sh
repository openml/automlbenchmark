#!/usr/bin/env bash
. $(dirname "$0")/../../setup/aliases.sh
if [[ -x "$(command -v apt-get)" ]]; then
    apt-get install -y build-essential swig
fi
PIP install --no-cache-dir -r requirements.txt
curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 PIP install
PIP install --no-cache-dir -r automl/frameworks/autosklearn/py_requirements.txt
