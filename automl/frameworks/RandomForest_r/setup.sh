#!/usr/bin/env bash
if ! [[ -x "$(command -v PIP)" ]]; then
    alias PIP=pip3
fi
f [[ -x "$(command -v apt-get)" ]]; then
    apt-get -y install software-properties-common apt-transport-https libxml2-dev
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E084DAB9
    add-apt-repository 'deb [arch=amd64,i386] https://cran.rstudio.com/bin/linux/ubuntu bionic-cran35/'
    apt update
    apt-get -y install r-base r-base-dev
fi
PIP install --no-cache-dir -r requirements.txt
#PIP install --no-cache-dir -r automl/frameworks/RandomForest_r/py_requirements.txt

Rscript -e 'install.packages(c("mlr", "mlrCPO", "ranger", "farff"), repos="https://cloud.r-project.org/")'
