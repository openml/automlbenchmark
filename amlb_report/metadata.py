from math import isnan

import openml as oml
import pandas as pd

from .util import Namespace


def dataset_metadata(task_id):
    # print(f"loading {task_id}")
    if task_id.startswith('openml.org'):
        tid = int(task_id.split("/")[2])
        return openml_metadata(tid)
    else:
        return file_metadata(task_id)


def to_int(v, default=-1):
    return default if isnan(v) else int(v)


def openml_metadata(tid):
    task = oml.tasks.get_task(task_id=tid, download_data=False)
    dataset = oml.datasets.get_dataset(task.dataset_id, download_data=False)
    did = dataset.dataset_id
    name = dataset.name
    dq = dataset.qualities
    nrows = to_int(dq['NumberOfInstances'])
    nrows_nas = to_int(dq['NumberOfInstancesWithMissingValues'])
    nnas = to_int(dq['NumberOfMissingValues'])
    nfeatures = to_int(dq['NumberOfFeatures'])
    nfeatures_numeric = to_int(dq['NumberOfNumericFeatures'])
    nfeatures_symbolic = to_int(dq['NumberOfSymbolicFeatures'])
    nfeatures_binary = to_int(dq['NumberOfBinaryFeatures'])
    nclasses = to_int(dq['NumberOfClasses'])
    # class_entropy = float(dq['ClassEntropy'])
    class_minsize = to_int(dq['MinorityClassSize'])
    class_majsize = to_int(dq['MajorityClassSize'])
    class_imbalance = float(dq['MajorityClassPercentage'])/float(dq['MinorityClassPercentage'])

    task_type = ('regression' if nclasses == 0
                 else 'binary' if nclasses == 2
                 else 'multiclass' if nclasses > 2
                 else 'unknown')
    # print(f"loaded {name}")
    return Namespace(
        task=f"openml.org/t/{tid}",
        dataset=f"openml.org/d/{did}",
        type=task_type,
        name=name,
        nrows=nrows,
        nrows_nas=nrows_nas,
        nnas=nnas,
        nfeatures=nfeatures,
        nfeatures_numeric=nfeatures_numeric,
        nfeatures_symbolic=nfeatures_symbolic,
        nfeatures_binary=nfeatures_binary,
        nclasses=nclasses,
        # class_entropy=class_entropy,
        class_minsize=class_minsize,
        class_majsize=class_majsize,
        class_imbalance=class_imbalance,
    )


def file_metadata(url):
    return Namespace(
        task=url,
        dataset=url,
        type='unknwown',
        name=url,
        nrows=-1,
        nfeatures=-1,
        nclasses=-1,
        # class_entropy=-1,
        class_imbalance=-1
    )


def load_dataset_metadata(results):
    tids = results.id.unique()
    # task names are hardcoded in benchmark definitions, so we need to map them with their task id
    lookup_df = results.filter(items=['id', 'task'], axis=1).drop_duplicates()
    lookup_map = {rec['id']: rec['task'] for rec in lookup_df.to_dict('records')}
    # print(lookup_map)
    metadata = {lookup_map[m.task]: m for m in [dataset_metadata(tid) for tid in tids]}
    return metadata


def render_metadata(metadata, filename='metadata.csv'):
    df = pd.DataFrame([m.__dict__ for m in metadata.values()], 
                      columns=['task', 'name', 'type', 'dataset', 
                               'nrows', 'nfeatures', 'nclasses',
                               'class_imbalance'])
    df.sort_values(by='name', inplace=True)
    if filename:
        df.to_csv(filename, index=False)
    return df
