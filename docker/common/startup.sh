#!/bin/bash

# http://wiki.bash-hackers.org/howto/getopts_tutorial
while getopts :c:t:d:f:a:s:p:m: opt; do
  case $opt in
    c)
	  start_call=$OPTARG;;
	t)
	  task_id=$OPTARG;;
	d)
	  datafile=$OPTARG;;
	f)
	  fold_n=$OPTARG;;
	a)
	  apikey=$OPTARG;;
	s)
	  time_s=$OPTARG;;
	p)
	  n_cores=$OPTARG;;
	m)
	  metric=$OPTARG;;
	\?)
	  echo "Invalid option: -$OPTARG";;
  esac	  
done

if [ -z ${start_call+x} ]; then echo "start_call is unset"; exit; fi
if [ -z ${fold_n+x} ]; then echo "fold_n is unset"; exit; fi
if [ -z ${apikey+x} ]; then echo "apikey is unset"; exit; fi
if [ -z ${time_s+x} ]; then echo "time_s is unset"; exit; fi
if [ -z ${n_cores+x} ]; then echo "n_cores is unset"; exit; fi
if [ -z ${metric+x} ]; then echo "metric is unset"; exit; fi

if [ ! -z ${task_id+x} ]; then
  /venvs/setup/bin/python3 ./common/load_data.py -t $task_id -f $fold_n -a $apikey --train ./common/train.arff --test ./common/test.arff
elif [ ! -z ${datafile+x} ]; then
  echo "No task was specified. Data on disk will be used."
  /venvs/setup/bin/python3 ./common/load_data.py -d /bench/automl/dataset -f $fold_n --train ./common/train.arff --test ./common/test.arff
else
  echo "Neither a task nor datafile was specified. Can not continue."; exit
fi

cd automl
$start_call $time_s $n_cores $metric
cd ..

/venvs/setup/bin/python3 ./common/evaluate.py ./common/test.arff ./common/predictions.csv $metric
