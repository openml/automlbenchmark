#!/usr/bin/env bash
VERSION=${1:-"latest"}
echo "setting up H2O version $VERSION"

HERE=$(dirname "$0")
. ${HERE}/../shared/setup.sh
if [[ -x "$(command -v apt-get)" ]]; then
    SUDO apt-get update
    SUDO apt-get install -y openjdk-8-jdk
fi
PIP install --no-cache-dir -r ${HERE}/requirements.txt

H2O_REPO="https://h2o-release.s3.amazonaws.com/h2o"

if  [[ "$VERSION" = "latest" ]]; then
    h2o_package="h2o"
elif [[ "$VERSION" = "zahradnik" ]]; then
    h2o_package="${H2O_REPO}/rel-zahradnik/4/Python/h2o-3.30.0.4-py2.py3-none-any.whl"
elif [[ "$VERSION" = "yule" ]]; then
    h2o_package="${H2O_REPO}/rel-yule/3/Python/h2o-3.28.1.3-py2.py3-none-any.whl"
elif [[ "$VERSION" = "yu" ]]; then
    h2o_package="${H2O_REPO}/rel-yu/4/Python/h2o-3.28.0.4-py2.py3-none-any.whl"
elif [[ "$VERSION" = "yau" ]]; then
    h2o_package="${H2O_REPO}/rel-yau/11/Python/h2o-3.26.0.11-py2.py3-none-any.whl"
elif [[ "$VERSION" = "yates" ]]; then
    h2o_package="${H2O_REPO}/rel-yates/5/Python/h2o-3.24.0.5-py2.py3-none-any.whl"
elif [[ "$VERSION" = "xu" ]]; then
    h2o_package="${H2O_REPO}/rel-xu/6/Python/h2o-3.22.1.6-py2.py3-none-any.whl"
elif [[ "$VERSION" = "xia" ]]; then
    h2o_package="${H2O_REPO}/rel-xia/5/Python/h2o-3.22.0.5-py2.py3-none-any.whl"
elif [[ "$VERSION" = "wright" ]]; then
    h2o_package="${H2O_REPO}/rel-wright/10/Python/h2o-3.20.0.10-py2.py3-none-any.whl"
elif [[ "$VERSION" = "nightly" ]]; then
    NIGHTLY=$(curl ${H2O_REPO}/master/latest)
    VERSION=$(curl ${H2O_REPO}/master/${NIGHTLY}/project_version)
    h2o_package="${H2O_REPO}/master/${NIGHTLY}/Python/h2o-${VERSION}-py2.py3-none-any.whl"
fi

if [[ -n "$h2o_package" ]]; then
    echo "installing H2O-3 $VERSION"
    PIP install --no-cache-dir -U ${h2o_package}
else
    echo "not installing any H2O release version"
fi

