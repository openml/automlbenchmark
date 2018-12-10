#!/usr/bin/env bash
VERSION=$1
echo "setting up H2O version $VERSION"

. $(dirname "$0")/../../setup/shared.sh
if [[ -x "$(command -v apt-get)" ]]; then
    apt-get install -y openjdk-8-jdk
fi
PIP install --no-cache-dir -r automl/frameworks/H2OAutoML/py_requirements.txt
if [[ "$VERSION" = "xia" || -z "$VERSION" ]]; then
    echo "installing xia"
    PIP install --no-cache-dir -U http://h2o-release.s3.amazonaws.com/h2o/rel-xia/2/Python/h2o-3.22.0.2-py2.py3-none-any.whl
elif [[ "$VERSION" = "wright" ]]; then
    echo "installing wright"
    PIP install --no-cache-dir -U http://h2o-release.s3.amazonaws.com/h2o/rel-wright/10/Python/h2o-3.20.0.10-py2.py3-none-any.whl
else
    echo "not installing any h2o release version"
fi