""" Saves the data of a specific OpenML task's fold to disk. """
import sys

import arff
import openml

output_path_train = 'train.csv'
output_path_test = 'test.csv'

task_id = int(sys.argv[1])
fold_no = int(sys.argv[2])

task = openml.tasks.get_task(task_id)
dataset_id = task.dataset_id
arff_path = "{}/{}/{}/dataset.arff".format(
    openml.config.cache_directory,
    openml.datasets.functions.DATASETS_CACHE_DIR_NAME,
    dataset_id
)

data_marker_read = False
data_rows = []
with open(arff_path, 'r') as arff_file:
    for line in arff_file:
        if '@DATA' in line:
            data_marker_read = True
            continue
        if data_marker_read:
            data_rows.append(line)

train_inds, test_inds = task.get_train_test_split_indices(fold=fold_no)

train_rows = [row for i, row in enumerate(data_rows) if i in train_inds]
test_rows = [row for i, row in enumerate(data_rows) if i in test_inds]

with open(output_path_train, 'w') as output_file:
    output_file.writelines(train_rows)

with open(output_path_test, 'w') as output_file:
    output_file.writelines(test_rows)
