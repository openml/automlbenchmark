#!/usr/bin/env bash
HERE=$(dirname "$0")
VERSION=${1:-"stable"}
REPO=${2:-"a-hanf/mlr3automl"}
MLR_REPO=${3:-"https://github.com/mlr-org"}

if [[ "$VERSION" == "latest" || "$VERSION" == "stable" ]]; then
    VERSION="master"
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
mkdir "${HERE}/r-packages/"

Rscript -e 'options(install.packages.check.source="no"); install.packages(c("mlr3", "mlr3pipelines", "mlr3misc", "mlr3oml", "mlr3hyperband", "mlr3tuning", "paradox"), repos="https://cloud.r-project.org/", lib="'"${HERE}/r-packages/"'")'
Rscript -e 'options(install.packages.check.source="no"); install.packages(c("remotes", "checkmate", "R6", "xgboost", "ranger", "LiblineaR", "emoa", "e1071", "glmnet"), repos="https://cloud.r-project.org/", lib="'"${HERE}/r-packages/"'")'
Rscript -e '.libPaths("'"${HERE}/r-packages/"'"); remotes::install_github("'"${MLR_REPO}"'/mlr3extralearners", lib="'"${HERE}/r-packages/"'")'
Rscript -e '.libPaths("'"${HERE}/r-packages/"'"); remotes::install_github("'"${REPO}"'", ref="'"${VERSION}"'", lib="'"${HERE}/r-packages/"'")'

Rscript -e '.libPaths("'"${HERE}/r-packages/"'"); packageVersion("mlr3automl")' | awk '{print $2}' | sed "s/[‘’]//g" >> "${HERE}/.installed"
echo "${VERSION}" >> "${HERE}/.installed"
