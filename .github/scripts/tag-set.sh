#!/usr/bin/env bash
TAG="$1"
PUSH=${2:-"no-push"}

function extract_version()
{
  local version=$(echo "$1" | sed -E 's/^(refs\/(heads|tags)\/)?(.*)/\3/')
  echo "$version"
}

NEW_TAG=$(extract_version $TAG)

git config user.name github-actions
git config user.email github-actions@github.com
git tag -f $NEW_TAG

HIGHEST_vTAG=$(extract_version $(git ls-remote --tags --sort=-v:refname origin | grep -P "/v\d.*" | head -n 1 | awk '{print $2}'))
#git fetch --all --tags
#LATEST_vTAG=$(git describe --contains `git rev-list --tags="v*" --max-count=1`)
#HIGHEST_vTAG=$(git tag --list 'v*' --sort=-v:refname | grep -P "v\d.*" | head -n 1)

echo "New tag: '$NEW_TAG', Highest version tag: '$HIGHEST_vTAG'"
if [ "$NEW_TAG" == "$HIGHEST_vTAG" ]; then
  echo "Additionally setting 'stable' tag"
  git tag -f stable
fi
if [[ "$PUSH" == "push" ]]; then
  git push --tags -f
fi
