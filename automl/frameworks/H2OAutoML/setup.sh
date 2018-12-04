#!/usr/bin/env bash
. ../setup/aliases.sh
if [[ -x "$(command -v apt-get)" ]]; then
    apt-get install -y openjdk-8-jdk
fi
PIP install --no-cache-dir -r requirements.txt
PIP install --no-cache-dir -r automl/frameworks/H2OAutoML/py_requirements.txt
PIP install -U http://h2o-release.s3.amazonaws.com/h2o/rel-xia/2/Python/h2o-3.22.0.2-py2.py3-none-any.whl
#PIP install -U https://s3.eu-central-1.amazonaws.com/sebp/builds/h2o-3.23.0.99999-py2.py3-none-any.whl
#PIP install -U ~/repos/h2o/h2o-3/h2o-py/build/dist/h2o-3.23.0.99999-py2.py3-none-any.whl