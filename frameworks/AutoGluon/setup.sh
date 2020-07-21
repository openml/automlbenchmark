#!/usr/bin/env bash
HERE=$(dirname "$0")

# creating local venv
. $HERE/../shared/setup.sh $HERE
#if [[ -x "$(command -v apt-get)" ]]; then
#    SUDO apt-get install -y libomp-dev
if [[ -x "$(command -v brew)" ]]; then
    brew install libomp
fi

cat $HERE/requirements.txt | sed '/^$/d' | while read -r i; do PIP install "$i"; done
#PIP install --no-cache-dir -r $HERE/requirements.txt
