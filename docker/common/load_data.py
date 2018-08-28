import argparse
from itertools import islice

import openml
from sklearn.model_selection import KFold

script_description = "Loads a dataset from disk or OpenML, and saves a train/test split of it to disk."
parser = argparse.ArgumentParser(description=script_description)
parser.add_argument('-t', '--task', default=None, type=int)
parser.add_argument('-d', '--datafile', default=None)
parser.add_argument('-f', '--fold', required=True, type=int)

# API Key below is a READ-only key from the OpenML R tutorial.
parser.add_argument('-a', '--apikey', default='c1994bdb7ecb3c6f3c8f3b35f4b47f1f')
parser.add_argument('--train', default='train.arff')
parser.add_argument('--test', default='test.arff')

parsed_args = parser.parse_args()

if parsed_args.datafile and parsed_args.task:
    print("Both a datafile and a task are specified. Will continue to use the OpenML task.")

if parsed_args.task:
    openml.config.apikey = parsed_args.apikey
    task = openml.tasks.get_task(parsed_args.task)
    dataset_id = task.dataset_id
    arff_path = "{}/{}/{}/dataset.arff".format(
        openml.config.get_cache_directory(),
        openml.datasets.functions.DATASETS_CACHE_DIR_NAME,
        dataset_id
    )
elif parsed_args.datafile:
    arff_path = parsed_args.datafile
else:
    raise ValueError("Either a datafile or a task should be specified.")


def read_attribute_name(line):
    """ Reads the attribute name for an attribute line in an ARFF file. """
    # Reference: https://www.cs.waikato.ac.nz/ml/weka/arff.html
    # @attribute <attribute-name> <datatype>
    # attribute-name may include spaces (but then MUST be quoted).
    _, attr_info = line.split(' ', 1)
    attr_name, attr_type = attr_info.split(' ', 1)
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

if parsed_args.task:
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
elif parsed_args.datafile:
    # For provided datasets, we assume that the target variable is already the last column.
    pass


if parsed_args.task:
    train_inds, test_inds = task.get_train_test_split_indices(fold=parsed_args.fold)
elif parsed_args.datafile:
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    train_inds, test_inds = next(islice(kfold.split(data_rows), parsed_args.fold, None))

train_rows = [row for i, row in enumerate(data_rows) if i in train_inds]
test_rows = [row for i, row in enumerate(data_rows) if i in test_inds]

with open(parsed_args.train, 'w') as output_file:
    output_file.writelines(header_lines)
    output_file.writelines(train_rows)

with open(parsed_args.test, 'w') as output_file:
    output_file.writelines(header_lines)
    output_file.writelines(test_rows)
