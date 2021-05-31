#!/usr/bin/env bash
HERE=$(dirname "$0")
VERSION=${1:-"stable"}
REPO=${2:-"https://github.com/a-hanf"}
MLR_REPO=${3:-"https://github.com/mlr-org"}

. $HERE/../shared/setup.sh "$HERE"
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

# Installing packages by default tries to write to a writeable folder for which you don't have write permissions.
# Add yourself to 'staff' usergroup
su usermod -a -G staff $USER

Rscript -e 'options(install.packages.check.source="no"); install.packages(c("mlr3", "mlr3pipelines", "mlr3misc", "mlr3oml", "mlr3hyperband", "mlr3tuning", "paradox"), repos="https://cloud.r-project.org/")'
Rscript -e 'options(install.packages.check.source="no"); install.packages(c("remotes", "checkmate", "R6", "xgboost", "ranger", "LiblineaR", "emoa", "e1071", "glmnet"), repos="https://cloud.r-project.org/")'
Rscript -e 'remotes::install_github("'"${MLR_REPO}"'/mlr3extralearners")'
Rscript -e 'remotes::install_github("'"${REPO}"'/mlr3automl")'
#Rscript -e 'remotes::install_github("'"${MLR_REPO}"'/mlr3pipelines")'
#Rscript -e 'remotes::install_github("'"${MLR_REPO}"'/mlr3oml")'
#Rscript -e 'remotes::install_github("'"${MLR_REPO}"'/paradox")'
#Rscript -e 'remotes::install_github("'"${MLR_REPO}"'/mlr3misc")'
#Rscript -e 'remotes::install_github("'"${MLR_REPO}"'/mlr3tuning@autotuner-notimeout")'
#Rscript -e 'remotes::install_github("'"${MLR_REPO}"'/mlr3@master")'
#Rscript -e 'remotes::install_github("'"${MLR_REPO}"'/mlr3hyperband@master")'

Rscript -e 'packageVersion("mlr3automl")' | awk '{print $2}' | sed "s/[‘’]//g" >> "${HERE}/.installed"
