#!/usr/bin/env bash
if ! [[ -x "$(command -v PIP)" ]]; then
    alias PIP=pip3
fi
if [[ -x "$(command -v apt-get)" ]]; then
    apt-get install -y build-essential swig
fi
PIP install --no-cache-dir -r requirements.txt
curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 PIP install
PIP install --no-cache-dir -r automl/frameworks/autosklearn/py_requirements.txt
