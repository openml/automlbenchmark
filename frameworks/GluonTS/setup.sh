#!/usr/bin/env bash

HERE=$(dirname "$0")
VERSION=${1:-"stable"}
REPO=${2:-"https://github.com/awslabs/gluonts.git"}
PKG=${3:-"gluonts"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="dev"
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

    # TOPIC: install readline-devel (for rpy2)
    echo "installing readline-devel..."
    SUDO  yum install -y readline readline-devel
    # ---------------------

    # TOPIC: install R
    echo "installing R..."
    SUDO  yum install -y gcc make sqlite-devel zlib-devel libffi-devel openssl-devel bzip2-devel wget tar gzip
    SUDO  yum install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
    SUDO  yum install -y R
    # R version=3.6.0
    SUDO  yum -y install libcurl-devel
    SUDO  R -e 'install.packages(c("forecast"), repos="https://cloud.r-project.org")'

elif [[ "${OS}" == "Ubuntu 18.04" || "${OS}" == "Ubuntu 20.04" ]]; then
    # TOPIC: update system
    echo "updating system..."
    DEBIAN_FRONTEND=noninteractive
    SUDO apt-get update -y
    SUDO apt-get upgrade -y
    SUDO apt-get install -y apt-utils make cmake build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
        libsqlite3-dev wget curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
        libffi-dev liblzma-dev libcurl4-openssl-dev p7zip-full awscli python3-dev
    # ------------------

    # TOPIC: install R
    echo "installing R..."
    # forecast requires R 3.6 (Which is not available for Ubuntu 18.04 by default.)
    if [[ "${OS}" == "Ubuntu 18.04" ]]; then
        SUDO apt install -y software-properties-common
        SUDO apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
        SUDO add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran35/'
        SUDO apt-get -y update
    fi
    apt install -y r-base
    SUDO R -e 'install.packages(c("forecast", "nnfor"), repos="https://cloud.r-project.org")'
    # ---------------------
fi

# TOPIC: installing orjson
echo "installing orjson..."
PIP install "orjson"
# -------------------------

if [[ "$VERSION" == "stable" ]]; then
    PIP install --no-cache-dir -U "${PKG}[mxnet,pro,Prophet,R]"
elif [[ "$VERSION" =~ ^[0-9] ]]; then
    PIP install --no-cache-dir -U "${PKG}[mxnet,pro,Prophet,R]==${VERSION}"
else
    TARGET_DIR="${HERE}/lib/${PKG}"
    rm -Rf ${TARGET_DIR}
    git clone --depth 1 --single-branch --branch ${VERSION} --recurse-submodules ${REPO} ${TARGET_DIR}
    cd ${TARGET_DIR}
    PY_EXEC_NO_ARGS="$(cut -d' ' -f1 <<<"$py_exec")"
    PY_EXEC_DIR=$(dirname "$PY_EXEC_NO_ARGS")
    env PATH="$PY_EXEC_DIR:$PATH"
    PIP install -e ".[mxnet,pro,Prophet,R]"
fi

PY -c "from gluonts import __version__; print(__version__)" >> "${HERE}/.setup/installed"
