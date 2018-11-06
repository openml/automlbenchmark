#!/usr/bin/env bash
pip3 install --no-cache-dir -r requirements.txt
#pip3 install --no-cache-dir -r automl/frameworks/RandomForest_r/py_requirements.txt

Rscript -e 'install.packages(c("mlr", "mlrCPO", "ranger", "farff"), repos="https://cloud.r-project.org/")'
