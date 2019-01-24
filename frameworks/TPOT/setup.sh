#!/usr/bin/env bash
HERE=$(dirname "$0")
. $HERE/../shared/setup.sh
PIP install --no-cache-dir -r $HERE/requirements.txt
