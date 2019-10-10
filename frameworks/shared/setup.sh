#!/usr/bin/env bash
#shopt -s expand_aliases

SUDO() {
  if [[ $EUID == 0 ]]; then
      "$@"
  else
      sudo "$@"
  fi
}

if [[ -x "$(command -v /venvs/bench/bin/pip3)" ]]; then
    pip_exec=/venvs/bench/bin/pip3
else
    pip_exec=pip3
fi

PIP() {
  $pip_exec "$@"
}

#if [[ -x "$(command -v /venvs/bench/bin/activate)" ]]; then
#    /venvs/bench/bin/activate
#fi
#command -v PIP
echo "PIP=$pip_exec"
PIP install --no-cache-dir -r requirements.txt
