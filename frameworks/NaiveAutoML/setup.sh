#!/usr/bin/env bash
HERE=$(dirname "$0")
VERSION=${1:-"stable"}
REPO=${2:-"https://github.com/fmohr/naiveautoml"}
PKG=${3:-"naiveautoml"}

echo "NaiveAutoML/setup.sh" "$@"

if [[ "$VERSION" == "latest" ]]; then
    VERSION="master"
fi

. ${HERE}/../shared/setup.sh ${HERE} true

PIP install -r ${HERE}/requirements.txt

# no __version__ available: https://github.com/fmohr/naiveautoml/issues/22
GET_VERSION_STABLE="import subprocess
import re
pip_list = subprocess.run('$pip_exec list'.split(), capture_output=True)
match = re.search(r'naiveautoml\s+([^\n]+)', pip_list.stdout.decode(), flags=re.IGNORECASE)
version, = match.groups()
print(version)"


if [[ "$VERSION" == "stable" ]]; then
    PIP install --no-cache-dir -U ${PKG}
    echo GET_VERSION_STABLE
    VERSION=$(PY -c "${GET_VERSION_STABLE}")
elif [[ "$VERSION" =~ ^[0-9] ]]; then
    PIP install --no-cache-dir -U ${PKG}==${VERSION}
else
  if [[ "$VERSION" =~ ^# ]]; then
    # Versions starting with a `#` are to be interpreted as commit hashes
    # The actual git clone command expects the hash without the `#` prefix.
    VERSION="${VERSION:1}"
  fi
  echo "Attempting to install from git+${REPO}.git@${VERSION}#egg=naiveautoml&subdirectory=python"
  PIP install -U "git+${REPO}.git@${VERSION}#egg=naiveautoml&subdirectory=python"
fi

echo $VERSION >> "${HERE}/.setup/installed"
