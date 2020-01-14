#!/usr/bin/env bash
VERSION=$1
echo "setting up H2O version $VERSION"

HERE=$(dirname "$0")
. $HERE/../shared/setup.sh
if [[ -x "$(command -v apt-get)" ]]; then
    SUDO apt-get update
    SUDO apt-get install -y openjdk-8-jdk
fi
PIP install --no-cache-dir -r $HERE/requirements.txt
if [[ "$VERSION" = "yau" || -z "$VERSION" ]]; then
    echo "installing H2O-3 yau"
    PIP install --no-cache-dir -U http://h2o-release.s3.amazonaws.com/h2o/rel-yau/11/Python/h2o-3.26.0.11-py2.py3-none-any.whl
elif [[ "$VERSION" = "yates" ]]; then
    echo "installing H2O-3 yates"
    PIP install --no-cache-dir -U http://h2o-release.s3.amazonaws.com/h2o/rel-yates/5/Python/h2o-3.24.0.5-py2.py3-none-any.whl
elif [[ "$VERSION" = "xu" ]]; then
    echo "installing H2O-3 xu"
    PIP install --no-cache-dir -U http://h2o-release.s3.amazonaws.com/h2o/rel-xu/6/Python/h2o-3.22.1.6-py2.py3-none-any.whl
elif [[ "$VERSION" = "xia" ]]; then
    echo "installing H2O-3 xia"
    PIP install --no-cache-dir -U http://h2o-release.s3.amazonaws.com/h2o/rel-xia/5/Python/h2o-3.22.0.5-py2.py3-none-any.whl
elif [[ "$VERSION" = "wright" ]]; then
    echo "installing H2O-3 wright"
    PIP install --no-cache-dir -U http://h2o-release.s3.amazonaws.com/h2o/rel-wright/10/Python/h2o-3.20.0.10-py2.py3-none-any.whl
else
    echo "not installing any H2O release version"
fi
