#!/bin/bash   
CALL_COMMAND=$1
TASK_ID=$2
TIME_LIMIT_SECONDS=$3
NUMBER_CORES=$4
API_KEY=$5

$CALL_COMMAND $TASK_ID $TIME_LIMIT_SECONDS $NUMBER_CORES $API_KEY > output.txt

echo "last output in file"
tail -1 output.txt