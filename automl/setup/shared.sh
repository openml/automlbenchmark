#!/usr/bin/env bash
shopt -s expand_aliases
if [[ -x "$(command -v /venvs/bench/bin/pip3)" ]]; then
    alias PIP='/venvs/bench/bin/pip3'
else
    alias PIP='pip3'
fi
#if [[ -x "$(command -v /venvs/bench/bin/activate)" ]]; then
#    /venvs/bench/bin/activate
#fi
echo "$(command -v PIP)"
echo "$(alias)"
PIP install --no-cache-dir -r requirements.txt
