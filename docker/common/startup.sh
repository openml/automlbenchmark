#!/bin/bash

# http://wiki.bash-hackers.org/howto/getopts_tutorial
while getopts :c:t:d:f:a:s:p:m:g: opt; do
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
	g)
	  allowedMemGB=$OPTARG;;
	\?)
	  echo "Invalid option: -$OPTARG";;
  esac	  
done

if [ -z ${start_call+x} ]; then echo "start_call is unset"; exit; fi
if [ -z ${fold_n+x} ]; then echo "fold_n is unset"; exit; fi
if [ -z ${time_s+x} ]; then echo "time_s is unset"; exit; fi
if [ -z ${n_cores+x} ]; then echo "n_cores is unset"; exit; fi
if [ -z ${metric+x} ]; then echo "metric is unset"; exit; fi

# We set the memory limit, if memory is overspecified we only give warnings, default is to leave 2GB for OS.
totalMemMB=$(free -m|awk '/^Mem:/{print $2}')
if [ ! -z ${allowedMemGB+x} ]; then
  allowedMemMB=$(( $allowedMemGB * 1024 ))
  if [ $allowedMemMB -gt $totalMemMB ]; then
    echo "WARNING: Specified memory exceeds system memory."
	echo "Specified: " $allowedMemMB "MB, Sytem: " $totalMemMB "MB."
  elif [ $allowedMemMB -gt $(( $totalMemMB - 2048 )) ]; then
    echo "WARNING: Specified memory within 2Gb of system memory."
	echo "We encourage a 2Gb buffer, because otherwise OS memory usage might interfere with the AutoML framework."
	echo "Specified: " $allowedMemMB "MB, Sytem: " $totalMemMB "MB."    
  fi
else
  allowedMemMB=$(( $totalMemMB - 2048 ))
  echo "Memory set to use all but 2GB (default): " $allowedMemMB "MB."
fi

if [ ! -z ${task_id+x} ]; then
  if [ ! -z ${apikey+x} ]; then
    /venvs/setup/bin/python3 -Wignore ./common/load_data.py -t $task_id -f $fold_n -a $apikey --train ./common/train.arff --test ./common/test.arff
  else
    /venvs/setup/bin/python3 -Wignore ./common/load_data.py -t $task_id -f $fold_n --train ./common/train.arff --test ./common/test.arff
  fi
elif [ ! -z ${datafile+x} ]; then
  echo "No task was specified. Data on disk will be used."
  /venvs/setup/bin/python3 -Wignore ./common/load_data.py -d /bench/dataset -f $fold_n --train ./common/train.arff --test ./common/test.arff
else
  echo "Neither a task nor datafile was specified. Can not continue."; exit
fi

cd automl
$start_call $time_s $n_cores $metric $allowedMemMB
cd ..

/venvs/setup/bin/python3 -Wignore ./common/evaluate.py ./common/test.arff ./common/predictions.csv $metric
