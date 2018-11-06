#!/usr/bin/env bash
pip3 install --no-cache-dir -r requirements.txt
curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip3 install
pip3 install --no-cache-dir -r automl/frameworks/autosklearn/py_requirements.txt
