#!/bin/bash

# This script could be used to prepare the datasets and tasks config for time-series forecast benchmark.

DATASETS_DIR=$1

# Assert that the datasets directory is provided
if [ -z "$DATASETS_DIR" ]; then
    echo "Please provide the datasets directory as an argument."
    exit 1
fi

# Get the directory where the script is located
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
DOWNLOAD_DATASETS_SCRIPT=$SCRIPT_DIR/download_datasets.py
GENERATE_TASK_CONFIGS_SCRIPT=$SCRIPT_DIR/generate_task_configs.py
GENERATE_ID_2_TASK_MAPPING_SCRIPT=$SCRIPT_DIR/generate_array_id_to_task_mapping.py
ROOT_DIR=$SCRIPT_DIR/../..

# Download M3C dataset if it does not exist
mkdir -p $DATASETS_DIR
if [ ! -f $DATASETS_DIR/M3C.xls ]; then
    wget https://forecasters.org/data/m3comp/M3C.xls -P $DATASETS_DIR
fi

### Start of python venv
VENV_NAME=temp_venv
if [ -d "$VENV_NAME" ]; then
    echo "Cleaning up existing virtual environment..."
    rm -rf $VENV_NAME
fi
python3 -m venv $VENV_NAME
source $VENV_NAME/bin/activate
pip install gluonts pandas orjson pyyaml xlrd awscli joblib

# Download datasets
python $DOWNLOAD_DATASETS_SCRIPT -d $DATASETS_DIR

# Generate tasks config (stored to default location: $HOME/.config/automlbenchmark/benchmarks)
python $GENERATE_TASK_CONFIGS_SCRIPT -d $DATASETS_DIR
python $GENERATE_ID_2_TASK_MAPPING_SCRIPT $HOME/.config/automlbenchmark/benchmarks

### End of python venv
deactivate
rm -rf $VENV_NAME