#!/usr/bin/env bash
pip3 install --no-cache-dir -r requirements.txt
pip3 install --no-cache-dir -r automl/frameworks/H2OAutoML/py_requirements.txt
#pip3 install -U http://h2o-release.s3.amazonaws.com/h2o/rel-xia/1/Python/h2o-3.22.0.1-py2.py3-none-any.whl
#pip3 install -U https://s3.eu-central-1.amazonaws.com/sebp/builds/h2o-3.23.0.99999-py2.py3-none-any.whl
pip3 install -U ~/repos/h2o/h2o-3/h2o-py/build/dist/h2o-3.23.0.99999-py2.py3-none-any.whl