#!/usr/bin/env bash
HERE=$(dirname "$0")
VERSION=${1:-"stable"}
WEKA=${2:-"https://prdownloads.sourceforge.net/weka/weka-3-8-5-azul-zulu-linux.zip"}

. ${HERE}/../shared/setup.sh "${HERE}"
if [[ -x "$(command -v apt-get)" ]]; then
    SUDO apt-get update
    SUDO apt-get install -y wget unzip openjdk-8-jdk
fi

if [[ "$VERSION" == "latest" || "$VERSION" == "stable" ]]; then
    URL="https://raw.githubusercontent.com/automl/autoweka/master"
    AUTOWEKA_ARCHIVE="autoweka-latest.zip"
else
    URL="http://www.cs.ubc.ca/labs/beta/Projects/autoweka"
    AUTOWEKA_ARCHIVE="autoweka-${VERSION}.zip"
fi

DOWNLOAD_DIR="$HERE/lib"
TARGET_DIR="$DOWNLOAD_DIR/autoweka"
rm -Rf "${TARGET_DIR}"

wget "$URL/$AUTOWEKA_ARCHIVE" -P "$DOWNLOAD_DIR"
unzip "$DOWNLOAD_DIR/$AUTOWEKA_ARCHIVE" -d "$TARGET_DIR"

if [[ "$VERSION" == "latest" || "$VERSION" == "stable" ]]; then
  # At some point after 2.6 the weka.jar is no longer included in the autoweka archive
  wget "$WEKA" -O "$DOWNLOAD_DIR/weka.zip"
  unzip "$DOWNLOAD_DIR/weka.zip" -d "$DOWNLOAD_DIR"
  # The weka.zip archive contains a folder named weka-VERSION.
  # By renaming it we can make the exec.py agnostic of the specific name.
  (cd "$DOWNLOAD_DIR" && find . -maxdepth 1 -type d -name "weka*" -exec mv {} weka \;)
fi
smac_dir=`find "$DOWNLOAD_DIR/autoweka" -maxdepth 1 -name "smac-*"`
chmod 755 "$smac_dir/smac"

find $HERE/lib/autoweka*.zip | sed -e 's/.*\/autoweka-\(.*\)\.zip/\1/' >> "${HERE}/.setup/installed"
