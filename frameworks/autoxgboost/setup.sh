#!/usr/bin/env bash
HERE=$(dirname "$0")
VERSION=${1:-"stable"}
REPO=${2:-"ja-thomas/autoxgboost"}
# currently both stable and latest maps to master branch
if [[ "$VERSION" == "latest" || "$VERSION" == "stable" ]]; then
    VERSION="master"
fi
. ${HERE}/../shared/setup.sh "${HERE}"
if [[ -x "$(command -v apt-get)" ]]; then
SUDO apt-get update
#SUDO apt-get install -y software-properties-common apt-transport-https libxml2-dev
#SUDO apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 51716619E084DAB9
#SUDO add-apt-repository 'deb [arch=amd64,i386] https://cran.rstudio.com/bin/linux/ubuntu bionic-cran35/'
SUDO apt-get install -y software-properties-common dirmngr
SUDO apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
SUDO add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
SUDO apt-get update
SUDO apt-get install -y r-base r-base-dev
SUDO apt-get install -y libgdal-dev libproj-dev
SUDO apt-get install -y libssl-dev libcurl4-openssl-dev
SUDO apt-get install -y libcairo2-dev libudunits2-dev
fi

mkdir "${HERE}/r-packages/"
Rscript -e 'options(install.packages.check.source="no"); install.packages(c("remotes", "mlr", "mlrMBO", "mlrCPO", "farff", "GenSA", "rgenoud", "xgboost"), repos="https://cloud.r-project.org/", lib="'"${HERE}/r-packages/"'")'
Rscript -e 'remotes::install_github("'"${REPO}"'", ref="'"${VERSION}"'", lib="'"${HERE}/r-packages/"'")'

Rscript -e '.libPaths("'"${HERE}/r-packages/"'"); packageVersion("autoxgboost")' | awk '{print $2}' | sed "s/[‘’]//g" >> "${HERE}/.installed"
