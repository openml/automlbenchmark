#!/bin/bash

let nargs=($# - 6)

start_call=${@: 1 :$nargs}
task_id=${@: -6:1}
fold_n=${@: -5:1}
apikey=${@: -4:1}
time_s=${@: -3:1}
n_cores=${@: -2:1}
metric=${@: -1:1}

/venvs/setup/bin/python3 ./common/load_data.py $task_id $fold_n $apikey ./common/train.arff ./common/test.arff

cd automl
$start_call $time_s $n_cores $metric
cd ..

/venvs/setup/bin/python3 ./common/evaluate.py ./common/test.arff ./common/predictions.csv $metric
