#!/usr/bin/env bash
HERE=$(dirname "$0")
. ${HERE}/../shared/setup.sh "$HERE"

if [[ -x "$(command -v apt-get)" ]]; then
    SUDO apt-get update
#    SUDO apt-get install -y software-properties-common apt-transport-https libxml2-dev
#    SUDO apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 51716619E084DAB9
#    SUDO add-apt-repository 'deb [arch=amd64,i386] https://cran.rstudio.com/bin/linux/ubuntu bionic-cran35/'
    SUDO apt-get install -y software-properties-common dirmngr
    SUDO apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
    SUDO add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
#    SUDO add-apt-repository ppa:c2d4u.team/c2d4u4.0+
    SUDO apt-get update
    SUDO apt-get install -y r-base r-base-dev
fi
#PIP install --no-cache-dir -r $HERE/requirements.txt
LIB="${HERE}/lib/"
mkdir "${LIB}"

Rscript -e 'options(install.packages.check.source="no"); install.packages(c("ranger", "mlr3", "mlr3learners", "mlr3pipelines", "farff"), repos="https://cloud.r-project.org/", lib="'"${LIB}"'")'
Rscript -e 'packageVersion("ranger")' | awk '{print $2}' | sed "s/[‘’]//g" >> "${HERE}/.setup/installed"
