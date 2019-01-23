#!/usr/bin/env bash
. $(dirname "$0")/../../setup/shared.sh
if [[ -x "$(command -v apt-get)" ]]; then
    SUDO apt-get -y install software-properties-common apt-transport-https libxml2-dev
    SUDO apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E084DAB9
    SUDO add-apt-repository 'deb [arch=amd64,i386] https://cran.rstudio.com/bin/linux/ubuntu bionic-cran35/'
    SUDO apt update
    SUDO apt-get -y install r-base r-base-dev
fi
#PIP install --no-cache-dir -r automl/frameworks/RandomForest_R/py_requirements.txt

SUDO Rscript -e 'install.packages(c("mlr", "mlrCPO", "ranger", "farff"), repos="https://cloud.r-project.org/")'
