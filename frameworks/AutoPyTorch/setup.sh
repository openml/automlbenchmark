#!/usr/bin/env bash

HERE=$(dirname "$0")
VERSION=${1:-"stable"}
REPO=${2:-"https://github.com/automl/Auto-PyTorch.git"}
PKG=${3:-"autoPyTorch"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="master"
fi

OS_NAME=$(cat /etc/os-release | grep '^NAME' | cut -d '"' -f 2)
OS_VERSION=$(cat /etc/os-release | grep '^VERSION_ID' | cut -d '"' -f 2)
OS="${OS_NAME} ${OS_VERSION}"
OS_SUPPORTED="Amazon Linux 2,Ubuntu 18.04,Ubuntu 20.04"

if echo "${OS_SUPPORTED}" | tr "," "\n" | grep -F -x -q "${OS}"; then
    echo "INFO: Operating system ${OS} is supported."
else
    echo "ERROR: Operating system ${OS} is not supported."
    exit 1
fi

# creating local venv
. ${HERE}/../shared/setup.sh ${HERE} true

PIP install --upgrade pip
PIP install --upgrade setuptools wheel

if [[ "${OS}" == "Amazon Linux 2" ]]; then
    # TOPIC: update system
    echo "updating system..."
    SUDO yum clean all
    SUDO rm -rf /var/cache/yum
    SUDO yum -y update

elif [[ "${OS}" == "Ubuntu 18.04" || "${OS}" == "Ubuntu 20.04" ]]; then
    # TOPIC: update system
    echo "updating system..."
    DEBIAN_FRONTEND=noninteractive
    SUDO apt-get update -y
    SUDO apt-get upgrade -y
    #SUDO apt-get install -y python3-opencv
fi

if [[ ${MODULE} == "timeseries" ]]; then
    echo "Info: AutoPyTorch: Installing forecasting extension."
    PKG_EXT="[forecasting]"
else
    echo "Info: AutoPyTorch: Installing without forecasting extension."
    PKG_EXT=""
fi
if [[ "$VERSION" == "stable" ]]; then
    PIP install --no-cache-dir -U "${PKG}${PKG_EXT}"
elif [[ "$VERSION" =~ ^[0-9] ]]; then
    PIP install --no-cache-dir -U "${PKG}${PKG_EXT}==${VERSION}"
else
    TARGET_DIR="${HERE}/lib/${PKG}"
    rm -Rf ${TARGET_DIR}
    git clone --depth 1 --single-branch --branch ${VERSION} --recurse-submodules ${REPO} ${TARGET_DIR}
    cd ${TARGET_DIR}
    PY_EXEC_NO_ARGS="$(cut -d' ' -f1 <<<"$py_exec")"
    PY_EXEC_DIR=$(dirname "$PY_EXEC_NO_ARGS")
    env PATH="$PY_EXEC_DIR:$PATH"
    PIP install -e ".${PKG_EXT}"
fi

PY -c "from autoPyTorch import __version__; print(__version__)" >> "${HERE}/.setup/installed"
