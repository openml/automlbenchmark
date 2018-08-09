""" Saves the data of a specific OpenML task's fold to disk.

Input:
 - OpenML Task ID
 - Fold number
 - OpenML API Key. This may be read-only.
 - Output path for train data.
 - Output path for test data.
"""
import sys

import openml

task_id = int(sys.argv[1])
fold_no = int(sys.argv[2])
api_key = sys.argv[3]
output_path_train = sys.argv[4]
output_path_test = sys.argv[5]

openml.config.apikey = api_key

task = openml.tasks.get_task(task_id)
dataset_id = task.dataset_id
arff_path = "{}/{}/{}/dataset.arff".format(
    openml.config.get_cache_directory(),
    openml.datasets.functions.DATASETS_CACHE_DIR_NAME,
    dataset_id
)


def read_attribute_name(line):
    """ Reads the attribute name for an attribute line in an ARFF file. """
    # Reference: https://www.cs.waikato.ac.nz/ml/weka/arff.html
    # @attribute <attribute-name> <datatype>
    # attribute-name may include spaces (but then MUST be quoted).
    _, attr_info = line.split(' ', 1)
    attr_name, attr_type = attr_info.rsplit(' ', 1)
    return attr_name.replace("'", "")


header_lines = []
attributes = []
data_rows = []
data_marker_read = False

with open(arff_path, 'r') as arff_file:
    for line in arff_file:
        if data_marker_read:
            data_rows.append(line)
        else:
            header_lines.append(line)

        if '@attribute' in line.lower():
            attributes.append(read_attribute_name(line))

        if '@data' in line.lower():
            data_marker_read = True

# We want to ensure that the last column is always the target as defined by the task so that AutoML systems do not
# need any additional information. For this, we need to order the data and attribute declarations accordingly.
target_index = attributes.index(task.target_name)

# If the last column is not the target column, we need to reorder the data.
if target_index != len(attributes) - 1:
    # We remove the new-line character before reshuffling the data so we do not have to keep track of where it is.
    data_rows = [line[:-1] for line in data_rows]
    data_rows_split = [row.split(',') for row in data_rows]
    reordered_data = [row[:target_index] + [row[-1]] + row[target_index+1:-1] + [row[target_index]]
                      for row in data_rows_split]
    data_rows = [",".join(row)+'\n' for row in reordered_data]

    last_attr_idx = [i for i, line in enumerate(header_lines) if '@attribute' in line.lower()][-1]
    target_attr_idx = [i for i, line in enumerate(header_lines) if '@attribute' in line.lower() and task.target_name in line][-1]
    header_lines[last_attr_idx], header_lines[target_attr_idx] = header_lines[target_attr_idx], header_lines[last_attr_idx]

train_inds, test_inds = task.get_train_test_split_indices(fold=fold_no)
train_rows = [row for i, row in enumerate(data_rows) if i in train_inds]
test_rows = [row for i, row in enumerate(data_rows) if i in test_inds]

with open(output_path_train, 'w') as output_file:
    output_file.writelines(header_lines)
    output_file.writelines(train_rows)

with open(output_path_test, 'w') as output_file:
    output_file.writelines(header_lines)
    output_file.writelines(test_rows)
