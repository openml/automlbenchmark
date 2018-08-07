import sys

import numpy as np
import openml
from sklearn.ensemble import RandomForestClassifier

task_id = sys.argv[1]
runtime_seconds = sys.argv[2]
number_cores = int(sys.argv[3])
openml.config.apikey = sys.argv[4]

print('Running RandomForest for task {} with a maximum time of {}s on {} cores.'
      .format(task_id, runtime_seconds, number_cores))

print('Downloading task.')
task = openml.tasks.get_task(task_id)

print('Running model on task.')
rfc = RandomForestClassifier(n_jobs=number_cores)
run = openml.runs.run_model_on_task(task, rfc)

# Because actual metric values are calculated on the server side of things, we actually need to do it manually here.
# Note that code below does not generalize to e.g. repeated CV.
print(np.mean(list(run.fold_evaluations[task.evaluation_measure][0].values())))



