FROM ubuntu:16.04

# == Python & Python OpenML setup ==
RUN apt-get update
RUN apt-get install -y python3.5 python-pip

# NumPy and SciPy need be installed, otherwise OpenML setup will fail.
RUN pip install --no-cache-dir numpy scipy
RUN pip install --no-cache-dir openml
# ===================================

WORKDIR /automl
ADD . /automl

# https://docs.docker.com/engine/reference/builder/#entrypoint
ENTRYPOINT ["/bin/bash", "run_your_code.sh", "python automl_scripts/randomforest.py"]