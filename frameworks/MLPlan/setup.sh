#!/usr/bin/env bash
HERE=$(dirname "$0")
VERSION=${1:-"latest"}

# creating local venv
. $HERE/../shared/setup.sh $HERE

if [[ -x "$(command -v apt-get)" ]]; then
    SUDO apt-get install -y wget unzip openjdk-11-jdk
fi

pip3 install --no-cache-dir -r $HERE/requirements.txt

MLPLAN_ARC="mlplan.zip"
DOWNLOAD_DIR="$HERE/lib"
TARGET_DIR="$DOWNLOAD_DIR/mlplan"

if [[ ! -e "$TARGET_DIR" ]]; then
    mkdir -p $DOWNLOAD_DIR
    echo "Download ML-Plan from extern"
    wget http://127.0.0.1/mlplan/version/$VERSION -O $DOWNLOAD_DIR/$MLPLAN_ARC
    echo "Download finished. Now unzip the downloaded file."
    unzip $DOWNLOAD_DIR/$MLPLAN_ARC -d $TARGET_DIR
fi
