#!/usr/bin/env bash
#shopt -s expand_aliases
echo "shared/setup.sh" "$@"
SHARED_DIR="$(cd $(dirname "${BASH_SOURCE[0]}") && pwd -P)"
MODULE_ROOT="$1"
APP_ROOT=$(dirname $(dirname "$SHARED_DIR"))
#APP_ROOT="$(pwd)"

SUDO() {
  if [[ $EUID == 0 ]]; then
      "$@"
  else
      sudo "$@"
  fi
}


if [[ -n "$MODULE_ROOT" ]]; then
    PY_VENV="$MODULE_ROOT/venv"
elif [[ -d "$APP_ROOT/venv" ]]; then
    PY_VENV="$APP_ROOT/venv"
fi

if [[ -n "$PY_VENV" ]]; then
    py_exec="$PY_VENV/bin/python"
    if [[ ! -x py_exec ]]; then
        python3 -m venv "$PY_VENV"
        $py_exec -m pip install -U pip wheel
    fi
    pip_exec="$py_exec -m pip"
    py_exec="$py_exec -W ignore"
else
    pip_exec="python3 -m pip"
    py_exec="python3 -W ignore"
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

PIP install --no-cache-dir -r $SHARED_DIR/requirements.txt
