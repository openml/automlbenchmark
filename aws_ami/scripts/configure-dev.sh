#!/bin/bash

echo "*** adding dev configuration ***"

echo "*** installing R ***"

apt-get update
apt-get install -y software-properties-common dirmngr
apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"

apt-get update
apt-get install -y r-base r-base-dev
apt-get install -y libgdal-dev libproj-dev
apt-get install -y libssl-dev libcurl4-openssl-dev
apt-get install -y libcairo2-dev libudunits2-dev
