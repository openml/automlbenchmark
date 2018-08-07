import sys

import numpy as np
import openml
from sklearn.model_selection import cross_val_score
from tpot import TPOTClassifier

if __name__ == '__main__':
    task_id = sys.argv[1]
    runtime_seconds = sys.argv[2]
    number_cores = int(sys.argv[3])
    openml.config.apikey = sys.argv[4]

    print('Running TPOT for task {} with a maximum time of {}s on {} cores.'
          .format(task_id, runtime_seconds, number_cores))

    print('Downloading task.')
    task = openml.tasks.get_task(task_id)
    X, y = task.get_X_and_y()

    print('Running model on task.')
    runtime_min = (int(runtime_seconds)/60) / 10
    runtime_min = 1
    tpot = TPOTClassifier(n_jobs=number_cores, population_size=10, generations=2, verbosity=2)

    repeats, folds, samples = task.get_split_dimensions()
    evaluation_measure = 'accuracy' if task.evaluation_measure == 'predictive_accuracy' else task.evaluation_measure
    scores = []
    if repeats == 1:
        cv_folds = [task.get_train_test_split_indices(fold=f, repeat=r) for r in range(repeats) for f in range(folds)]
        for (train_idx, test_idx) in cv_folds:
            try:
                tpot.fit(X[train_idx, :], y[train_idx])
            except KeyboardInterrupt:
                pass
            scores.append(tpot.score(X[test_idx, :], y[test_idx]))

        # Because actual metric values are calculated on the server side of things, we actually need to do it manually here.
        # Note that code below does not generalize to e.g. repeated CV.
        print(np.mean(scores))
    else:
        print('no output.')
