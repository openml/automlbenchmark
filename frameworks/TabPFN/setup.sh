#!/usr/bin/env bash
HERE=$(dirname "$0")

# read secret_url from a secret_link.txt
SECRET_URL=$(cat "${HERE}/secret_link.txt")

. "${HERE}/../shared/setup.sh" "${HERE}" true

PIP install --no-cache-dir -U ${SECRET_URL}

PY -c "from importlib.metadata import version; print(version('tabpfn'))" >> "${HERE}/.setup/installed"
