#!/usr/bin/env bash
. $(dirname "$0")/../../setup/shared.sh
if [[ -x "$(command -v apt-get)" ]]; then
    apt-get install -y build-essential swig
fi
curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | sed '/^$/d' | while read -r i; do PIP install "$i"; done
PIP install --no-cache-dir -r automl/frameworks/autosklearn/py_requirements.txt
