#!/usr/bin/env bash
HERE=$(dirname "$0")
AMLB_DIR="$1"
VERSION=${2:-"latest"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="master"
fi

# creating local venv
. $HERE/../shared/setup.sh $HERE

#curl "https://raw.githubusercontent.com/EpistasisLab/tpot/${VERSION}/requirements.txt" | sed '/^$/d' | while read -r i; do PIP install "$i"; done
PIP install --no-cache-dir -r "https://raw.githubusercontent.com/EpistasisLab/tpot/${VERSION}/optional-requirements.txt"
#PIP install --no-cache-dir -r $HERE/requirements.txt

if [[ "$VERSION" =~ ^[0-9] ]]; then
    PIP install --no-cache-dir -U tpot[dask]==${VERSION}
else
    PIP install --no-cache-dir -U -e git+https://github.com/EpistasisLab/tpot.git@${VERSION}#egg=tpot
fi
