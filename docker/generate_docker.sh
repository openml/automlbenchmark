#!/bin/bash   
DIRECTORY=$1

# Copy the first layers.
cat DockerHeader > $DIRECTORY/Dockerfile

# Copy the AutoML-specific layers.
cat $DIRECTORY/CustomDockerCode >> $DIRECTORY/Dockerfile
echo '\n' >> $DIRECTORY/Dockerfile

# Add the AutoML system
echo "ADD $DIRECTORY /bench/automl" >> $DIRECTORY/Dockerfile

if [ $# -eq 2 ]; then
	echo "Adding dataset to docker file"
	echo "ADD $2 /bench/dataset" >> $DIRECTORY/Dockerfile
fi

# Append the ENTRY_POINT call.
cat DockerFooter >> $DIRECTORY/Dockerfile
