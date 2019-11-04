#!/usr/bin/env bash
#shopt -s expand_aliases

MODULE_ROOT="$1"

SUDO() {
  if [[ $EUID == 0 ]]; then
      "$@"
  else
      sudo "$@"
  fi
}


if [[ -n "$MODULE_ROOT" ]]; then
    PY_VENV="$MODULE_ROOT/venv"
elif [[ -d "/venvs/bench" ]]; then
    PY_VENV="/venvs/bench"
fi

if [[ -n "$PY_VENV" ]]; then
    py_exec="$PY_VENV/bin/python"
    if [[ ! -f py_exec ]]; then
        python3 -m venv "$PY_VENV"
        $py_exec -m pip install -U pip
    fi
    py_exec="$py_exec -W ignore"
    pip_exec="$PY_VENV/bin/pip"
else
    pip_exec=pip3
    py_exec=python3
fi

PY() {
  $py_exec "$@"
}

PIP() {
  $pip_exec "$@"
}

#if [[ -x "$(command -v $PY_VENV/bin/activate)" ]]; then
#    $PY_ROOT/activate
#fi

#echo "PY=$(command -v PY)"
#echo "PIP=$(command -v PIP)"
echo "PY=$py_exec"
echo "PIP=$pip_exec"
