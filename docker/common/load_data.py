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

with open(arff_path, 'r') as arff_file:
    arff_content = arff.load(arff_file)

dataset = arff_content['data']
train_inds, test_inds = task.get_train_test_split_indices(fold=fold_no)

train_data = [row for i, row in enumerate(dataset) if i in train_inds]
test_data = [row for i, row in enumerate(dataset) if i in test_inds]

with open(output_path_train, 'w') as output_file:
    for row in train_data:
        output_file.write(','.join([str(el) for el in row])+'\n')

with open(output_path_test, 'w') as output_file:
    for row in test_data:
        output_file.write(','.join([str(el) for el in row])+'\n')
