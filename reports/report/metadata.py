import math

import openml as oml
import pandas as pd

from .util import Namespace


def dataset_metadata(task_id):
    # print(f"loading {task_id}")
    tid = int(task_id.split("/")[2]) if task_id.startswith('openml.org') else int(task_id)
    task = oml.tasks.get_task(task_id=tid, download_data=False)
    dataset = oml.datasets.get_dataset(task.dataset_id, download_data=False)
    did = dataset.dataset_id
    name = dataset.name
    dq = dataset.qualities
    features = list(dataset.features.values())
    feature_types = oml.datasets.OpenMLDataFeature.LEGAL_DATA_TYPES

    features_excluded = []
    if dataset.row_id_attribute:
        features_excluded.append(dataset.row_id_attribute)
    if dataset.ignore_attribute:
        features_excluded.extend(dataset.ignore_attribute)

    nrows = int(dq['NumberOfInstances'])
    nfeatures = int(dq['NumberOfFeatures'])
    nclasses = int(dq['NumberOfClasses'])

    features_stats = {}
    for ft in feature_types:
        feats = [f for f in features if f.data_type == ft and f.name not in features_excluded]
        missing = [f.number_missing_values for f in feats]
        features_stats[ft] = dict(
            count=len(feats),
            na_min=min(missing, default=math.nan),
            na_max=max(missing, default=math.nan),
        )
        if ft == 'nominal':
            distinct = [len(f.nominal_values) for f in feats]
            features_stats[ft].update(
                distinct_min=min(distinct, default=math.nan),
                distinct_max=max(distinct, default=math.nan)
            )

    flat_feature_stats = {f"{ft}_{k}": v
                          for ft, k, v in [(ft, k, v)
                                           for ft, stats in features_stats.items()
                                           for k, v in stats.items()]}

    # class_entropy = float(dq['ClassEntropy'])
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
        nfeatures=nfeatures,
        nclasses=nclasses,
        # class_entropy=class_entropy,
        class_imbalance=class_imbalance,
    ).extend(**flat_feature_stats)


def load_dataset_metadata(results):
    tids = results.id.unique()
    # task names are hardcoded in benchmark definitions, so we need to map them with their task id
    lookup_df = results.filter(items=['id', 'task'], axis=1).drop_duplicates()
    lookup_map = {rec['id']: rec['task'] for rec in lookup_df.to_dict('records')}
    # print(lookup_map)
    metadata = {lookup_map[m.task]: m for m in [dataset_metadata(tid) for tid in tids]}
    return metadata


def render_metadata(metadata, filename='metadata.csv'):
    fixed_cols = ['task', 'name', 'type', 'dataset',
                  'nrows', 'nfeatures', 'nclasses',
                  'class_imbalance']
    dyn_cols = [k for k in next(iter(metadata.values())).__dict__.keys() if k not in fixed_cols] if len(metadata.values()) > 0 else []
    dyn_cols.sort()
    df = pd.DataFrame([m.__dict__ for m in metadata.values()],
                      columns=[]+fixed_cols+dyn_cols)
    df.sort_values(by='name', inplace=True)
    if filename:
        df.to_csv(filename, index=False)
    return df
