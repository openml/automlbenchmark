#!/usr/bin/env bash
HERE=$(dirname "$0")
VERSION=${1:-"stable"}
REPO=${2:-"a-hanf/mlr3automl"}
MLR_REPO=${3:-"https://github.com/mlr-org"}

# Version can be specified as 'stable', 'latest', a branch or a commit hash (indicated by starting with '#')
if [[ "$VERSION" == "latest" || "$VERSION" == "stable" ]]; then
  VERSION="master"
fi

if [[ "$VERSION" =~ ^# ]]; then
  VERSION="${VERSION:1}"
else
  # if VERSION is not a hash, it should be a branch (or a format which is not (officially) supported)
  COMMIT=$(git ls-remote "https://github.com/${REPO}" | grep "refs/heads/${VERSION}" | cut -f 1)
  if [[ -z $COMMIT ]]; then
    echo "Could not resolve version ${VERSION}. It is not a branch on https://github.com/${REPO}."
    echo "Continuing setup, install_github will try to resolve 'ref=${VERSION}'."
  else
    VERSION=$COMMIT
  fi
fi

. $HERE/../shared/setup.sh "$HERE"
if [[ -x "$(command -v apt-get)" ]]; then
  SUDO apt-get update
  SUDO apt-get install -y software-properties-common dirmngr
  SUDO apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
  SUDO add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
  SUDO apt-get update
  SUDO apt-get install -y r-base r-base-dev
  SUDO apt-get install -y libgdal-dev libproj-dev
  SUDO apt-get install -y libssl-dev libcurl4-openssl-dev
  SUDO apt-get install -y libcairo2-dev libudunits2-dev
fi

# We install the packages to a subdirectory in the framework folder, similar to venvs for Python, because:
# - we want to be able to install different package versions for different frameworks in one local installation
# - the default package directory is not always writeable (e.g. on Github CI)
LIB="${HERE}/lib/"
mkdir "${LIB}"

Rscript -e 'options(install.packages.check.source="no"); install.packages("remotes", repos="https://cloud.r-project.org/", lib="'"${LIB}"'")'
Rscript -e 'options(install.packages.check.source="no"); install.packages(c("xgboost", "ranger", "LiblineaR", "emoa", "e1071", "glmnet"), repos="https://cloud.r-project.org/", lib="'"${LIB}"'")'
Rscript -e '.libPaths("'"${LIB}"'"); remotes::install_github("mlr-org/mlr3learners", dependencies=TRUE)'
Rscript -e '.libPaths("'"${LIB}"'"); remotes::install_github("mlr-org/mlr3extralearners")'
Rscript -e '.libPaths("'"${LIB}"'"); remotes::install_github("'"${REPO}"'", lib="'"${LIB}"'", dependencies=TRUE)'

OFFICIAL_VERSION=$(Rscript -e '.libPaths("'"${LIB}"'"); packageVersion("mlr3automl")' | awk '{print $2}' | sed "s/[‘’]//g")
echo "${OFFICIAL_VERSION}#${VERSION:0:7}" >> "${HERE}/.setup/installed"
