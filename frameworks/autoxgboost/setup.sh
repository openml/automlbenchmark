#!/usr/bin/env bash
HERE=$(dirname "$0")
. $HERE/../shared/setup.sh
if [[ -x "$(command -v apt-get)" ]]; then
    SUDO apt-get update
    SUDO apt-get install -y software-properties-common apt-transport-https libxml2-dev
    SUDO apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E084DAB9
    SUDO add-apt-repository 'deb [arch=amd64,i386] https://cran.rstudio.com/bin/linux/ubuntu bionic-cran35/'
    SUDO apt-get update
    SUDO apt-get install -y r-base r-base-dev

    SUDO apt-get install default-jre
    SUDO apt-get install default-jdk
    SUDO R CMD javareconf
    SUDO apt-get install r-cran-rjava
    SUDO apt-get install libgdal-dev libproj-dev

    SUDO apt-get install libssl-dev
    SUDO apt-get install libcurl4-openssl-dev
    SUDO apt-get install libcairo2-dev
    SUDO apt-get install libudunits2-dev
fi
#PIP install --no-cache-dir -r $HERE/requirements.txt

SUDO Rscript -e 'options(install.packages.check.source="no"); install.packages(c("devtools", "mlr", "mlrMBO", "mlrCPO", "farff", "xgboost"), repos="https://cloud.r-project.org/", dependencies = TRUE)'
SUDO Rscript -e 'devtools::install_github("ja-thomas/autoxgboost")'

