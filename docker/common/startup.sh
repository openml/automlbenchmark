#!/bin/bash

let nargs=($# - 5)

start_call=${@: 1 :$nargs}
task_id=${@: -5:1}
fold_n=${@: -4:1}
time_s=${@: -3:1}
n_cores=${@: -2:1}
apikey=${@: -1:1}

python3 ./common/load_data.py $task_id $fold_n
cd automl
$start_call $task_id $time_s $n_cores $apikey
