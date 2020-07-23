#!/usr/bin/env bash
HERE=$(dirname "$0")
. $HERE/../shared/setup.sh
if [[ -x "$(command -v apt-get)" ]]; then
    echo "Setup system dependencies"
    SUDO apt-get update
    SUDO apt-get install -y wget unzip openjdk-11-jdk
fi

MLPLAN_ARC="mlplan.zip"
DOWNLOAD_DIR="$HERE/lib"
TARGET_DIR="$DOWNLOAD_DIR/mlplan"
if [[ ! -e "$TARGET_DIR" ]]; then
    mkdir -p $DOWNLOAD_DIR
    echo "Download ML-Plan from extern"
    wget http://192.168.2.105/mlplan/latest/ -O $DOWNLOAD_DIR/$MLPLAN_ARC
    echo "Download finished. Now unzip the downloaded file."
    unzip $DOWNLOAD_DIR/$MLPLAN_ARC -d $TARGET_DIR
fi
