#!/bin/bash   
DIRECTORY=$1

# Copy the first layers.
cat DockerHeader > $DIRECTORY/Dockerfile

# Add the AutoML system
echo "ADD $DIRECTORY /bench/automl" >> $DIRECTORY/Dockerfile

# Copy the AutoML-specific layers.
cat $DIRECTORY/CustomDockerCode >> $DIRECTORY/Dockerfile

# Append the ENTRY_POINT call.
cat DockerFooter >> $DIRECTORY/Dockerfile