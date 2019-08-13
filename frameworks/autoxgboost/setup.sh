#!/usr/bin/env bash

shopt -s expand_aliases
if [[ -x "$(command -v /venvs/bench/bin/pip3)" ]]; then
    alias PIP='/venvs/bench/bin/pip3'
else
    alias PIP='pip3'
fi

if [[ $EUID == 0 ]]; then
    alias SUDO=''
else
    alias SUDO='sudo'
fi

#if [[ -x "$(command -v /venvs/bench/bin/activate)" ]]; then
#    /venvs/bench/bin/activate
#fi
echo "$(command -v PIP)"
PIP install --no-cache-dir -r requirements.txt

if [[ -x "$(command -v apt-get)" ]]; then
    SUDO apt-get update
    SUDO apt-get install -y software-properties-common apt-transport-https libxml2-dev
    SUDO apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E084DAB9
    SUDO add-apt-repository 'deb [arch=amd64,i386] https://cran.rstudio.com/bin/linux/ubuntu bionic-cran35/'
    SUDO apt-get update
    SUDO apt-get install -y r-base r-base-dev
fi
#PIP install --no-cache-dir -r $HERE/requirements.txt

SUDO Rscript -e 'options(install.packages.check.source="no"); install.packages(c("devtools", "mlr", "mlrMBO", "mlrCPO", "farff", "xgboost"), repos="https://cloud.r-project.org/", dependencies = TRUE)'
SUDO Rscript -e 'devtools::install_github("ja-thomas/autoxgboost")'

