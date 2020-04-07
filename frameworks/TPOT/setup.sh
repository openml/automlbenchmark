#!/usr/bin/env bash
VERSION=${1:-"v0.11.1"}
HERE=$(dirname "$0")
. $HERE/../shared/setup.sh $HERE
curl "https://raw.githubusercontent.com/EpistasisLab/tpot/${VERSION}/requirements.txt" | sed '/^$/d' | while read -r i; do PIP install "$i"; done
PIP install --no-cache-dir -r $HERE/requirements.txt
