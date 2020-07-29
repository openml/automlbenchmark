#!/usr/bin/env bash
HERE=$(dirname "$0")
VERSION=${2:-"latest"}

echo "Setup ML-Plan for version $VERSION"

# creating local venv
. $HERE/../shared/setup.sh $HERE

if [[ -x "$(command -v apt-get)" ]]; then
    echo "setup system packages"
    SUDO apt-get install -y wget unzip openjdk-11-jdk
    SUDO apt-get install -y libatlas3-base libopenblas-base
fi

PIP install --no-cache-dir -r $HERE/requirements.txt

MLPLAN_ARC="mlplan.zip"
DOWNLOAD_DIR="$HERE/lib"
TARGET_DIR="$DOWNLOAD_DIR/mlplan"

if [[ ! -e "$TARGET_DIR" ]]; then
    mkdir -p $DOWNLOAD_DIR
    echo "Download ML-Plan from extern"
    wget https://download.mlplan.org/version/$VERSION -O $DOWNLOAD_DIR/$MLPLAN_ARC
    echo "Download finished. Now unzip the downloaded file."
    unzip $DOWNLOAD_DIR/$MLPLAN_ARC -d $TARGET_DIR
fi
