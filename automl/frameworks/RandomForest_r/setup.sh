#!/usr/bin/env bash
. $(dirname "$0")/../../setup/shared.sh
if [[ -x "$(command -v apt-get)" ]]; then
    apt-get -y install software-properties-common apt-transport-https libxml2-dev
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E084DAB9
    add-apt-repository 'deb [arch=amd64,i386] https://cran.rstudio.com/bin/linux/ubuntu bionic-cran35/'
    apt update
    apt-get -y install r-base r-base-dev
fi
#PIP install --no-cache-dir -r automl/frameworks/RandomForest_R/py_requirements.txt

Rscript -e 'install.packages(c("mlr", "mlrCPO", "ranger", "farff"), repos="https://cloud.r-project.org/")'
