#!/usr/bin/env bash
shopt -s expand_aliases
HERE=$(dirname "$0")

. "$HERE/.setup/setup_env"
. "$AMLB_ROOT/frameworks/shared/setup.sh" "$HERE" true
PIP install -r "$HERE/requirements.txt"

PY -c "from sklearn import __version__; print(__version__)" >> "${HERE}/.installed"
