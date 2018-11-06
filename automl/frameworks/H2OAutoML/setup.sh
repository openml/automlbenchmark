#!/usr/bin/env bash
pip3 install --no-cache-dir -r requirements.txt
pip3 install --no-cache-dir -r automl/frameworks/H2OAutoML/py_requirements.txt
pip3 install http://h2o-release.s3.amazonaws.com/h2o/master/4402/Python/h2o-3.21.0.4402-py2.py3-none-any.whl
#pip3 install https://s3.eu-central-1.amazonaws.com/sebp/builds/h2o-3.21.0.99999-py2.py3-none-any.whl
