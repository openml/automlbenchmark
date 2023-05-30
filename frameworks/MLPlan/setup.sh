#!/usr/bin/env bash
HERE=$(dirname "$0")
VERSION=${1:-"stable"}
if [[ "$VERSION" == "stable" ]]; then
    VERSION="latest"
fi

echo "Setup ML-Plan for version $VERSION"

# creating local venv
. ${HERE}/../shared/setup.sh ${HERE} true

if [[ -x "$(command -v apt-get)" ]]; then
    echo "setup system packages"
    SUDO apt-get update
    SUDO apt-get install -y wget unzip openjdk-11-jdk
    SUDO apt-get install -y libatlas3-base libopenblas-base
fi

PIP install --no-cache-dir -r $HERE/requirements.txt

MLPLAN_ARC="mlplan.zip"
DOWNLOAD_DIR="$HERE/lib"
TARGET_DIR="$DOWNLOAD_DIR/mlplan"
rm -Rf ${TARGET_DIR}

mkdir -p $DOWNLOAD_DIR
echo "Download ML-Plan from extern"
wget -q --no-check-certificate https://download.mlplan.org/version/$VERSION -O $DOWNLOAD_DIR/$MLPLAN_ARC
echo "Download finished. Now unzip the downloaded file."
unzip $DOWNLOAD_DIR/$MLPLAN_ARC -d $TARGET_DIR

find $HERE/lib/mlplan/*.jar | sed -e 's/.*\/mlplan-cli-\(.*\)\.jar/\1/' >> "${HERE}/.setup/installed"
