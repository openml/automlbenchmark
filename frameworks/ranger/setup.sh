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
fi
#PIP install --no-cache-dir -r $HERE/requirements.txt

SUDO Rscript -e 'options(install.packages.check.source="no"); install.packages(c("mlr", "mlrCPO", "ranger", "farff"), repos="https://cloud.r-project.org/")'
