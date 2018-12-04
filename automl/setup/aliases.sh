#!/usr/bin/env bash
if [[ -x "$(command -v /venvs/bench/bin/pip3)" ]]; then
    alias PIP='/venvs/bench/bin/pip3'
else
    alias PIP='pip3'
fi
