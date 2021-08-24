#!/usr/bin/env bash
HERE=$(dirname "$0")
VERSION=${1:-"stable"}
H2O_REPO=${2:-"https://h2o-release.s3.amazonaws.com/h2o"}
echo "setting up H2O version $VERSION"

. ${HERE}/.setup/setup_env
. ${HERE}/../shared/setup.sh ${HERE} true
if [[ -x "$(command -v apt-get)" ]]; then
    SUDO apt-get update
    SUDO apt-get install -y openjdk-8-jdk
fi
PIP install --no-cache-dir -r ${HERE}/requirements.txt

if [[ -n "$H2O_WHL" ]]; then
    h2o_package="${H2O_WHL}"
elif  [[ "$VERSION" = "stable" ]]; then
    h2o_package="h2o"
elif [[ "$VERSION" = "latest" ]]; then
    NIGHTLY=$(curl ${H2O_REPO}/master/latest)
    VERSION=$(curl ${H2O_REPO}/master/${NIGHTLY}/project_version)
    h2o_package="${H2O_REPO}/master/${NIGHTLY}/Python/h2o-${VERSION}-py2.py3-none-any.whl"
elif [[ "$VERSION" =~ ^[0-9] ]]; then
    h2o_package="h2o==${VERSION}"
fi

if [[ -n "$h2o_package" ]]; then
    echo "installing H2O-3 $VERSION"
    PIP install --no-cache-dir --force-reinstall -U ${h2o_package}
else
    echo "not installing any H2O release version"
fi

PY -c "from h2o import __version__; print(__version__)" | grep "^[[:digit:]]\." >> "${HERE}/.setup/installed"

