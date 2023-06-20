#!/usr/bin/env bash
VERSION="$1"
PUSH=${2:-"no-push"}
ADD_COMMIT=${3:-"add-commit"}

VERSION_FILE='amlb/__version__.py'
VERSION_PLACEHOLDER='__version__ =\s+"(.*)"$'
DEV_PLACEHOLDER='_dev_version =\s+"(.*)"$'

function get_version()
{
  local version=$(grep -P "$2" "$1" | sed -E "s/$2/\1/")
  echo "$version"
}

function set_version()
{
  sed -i -E "s/$2/__version__ = \"$3\"/g" "$1"
  echo $(get_version "$1" "$2")
}

OLD_VERSION=$(get_version "$VERSION_FILE" "$VERSION_PLACEHOLDER")
DEV_VERSION=$(get_version "$VERSION_FILE" "$DEV_PLACEHOLDER")
NEW_VERSION=$(echo ${VERSION:-"$DEV_VERSION"} | sed -E 's/^v([0-9].*)/\1/')
echo "setting version from \"$OLD_VERSION\" to \"$NEW_VERSION\""
FINAL_VERSION=$(set_version "$VERSION_FILE" "$VERSION_PLACEHOLDER" "$NEW_VERSION")
echo "version changed to \"$FINAL_VERSION\""
echo "VERSION=$FINAL_VERSION" >> $GITHUB_ENV

if [[ "$PUSH" == "push" ]]; then
  if [[ -x "$(command -v git)" ]]; then
    git config user.name github-actions
    git config user.email github-actions@github.com
    if [[ "$ADD_COMMIT" == "add-commit" ]]; then
      git commit -am "Update version to $FINAL_VERSION"
      git push
    else
      git commit -a --amend --no-edit
      git push -f
    fi
  else
    echo "Can not push the version changes as git is not available."
    exit 1
  fi
fi
