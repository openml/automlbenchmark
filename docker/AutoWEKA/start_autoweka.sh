#!/bin/bash
time_s=$1
n_cores=$2
metric=$3
let time_m=$time_s/60

if [ $metric == acc ] ; then
  metric=errorRate
elif [ $metric == auc ] ; then
  metric=areaUnderROC
elif [ $metric == logloss ] ; then
  metric=kBInformation
fi

echo "TODO: set memlimit"
java -cp autoweka.jar weka.classifiers.meta.AutoWEKAClassifier -t ../common/train.arff -T ../common/test.arff -memLimit 1024 -classifications "weka.classifiers.evaluation.output.prediction.CSV -distribution -file predictions.csv" -timeLimit $time_m -parallelRuns $n_cores -metric $metric

python3 reformat_output.py predictions.csv ../common/predictions.csv
