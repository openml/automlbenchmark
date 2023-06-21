#!/usr/bin/env bash
HERE=$(dirname "$0")
VERSION=${1:-"stable"}
REPO=${2:-"https://github.com/fmohr/naiveautoml"}
PKG=${3:-"naiveautoml"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="master"
fi

. ${HERE}/../shared/setup.sh ${HERE} true

PIP install -r ${HERE}/requirements.txt
if [[ "$VERSION" == "stable" ]]; then
    PIP install --no-cache-dir -U ${PKG}
elif [[ "$VERSION" =~ ^[0-9] ]]; then
    PIP install --no-cache-dir -U ${PKG}==${VERSION}
else
    PIP install -U "git+${REPO}.git@${VERSION}#egg=naiveautoml&subdirectory=python"
fi

# no __version__ available: https://github.com/fmohr/naiveautoml/issues/22
GET_VERSION="import subprocess
import re
pip_list = subprocess.run('$pip_exec list'.split(), capture_output=True)
match = re.search(r'naiveautoml\s+([^\n]+)', pip_list.stdout.decode(), flags=re.IGNORECASE)
version, = match.groups()
print(version)"

echo $GET_VERSION

PY -c "${GET_VERSION}" >> "${HERE}/.setup/installed"
