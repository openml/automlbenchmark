"""
**openml** module implements the abstractions defined in **data** module
to expose `OpenML<https://www.openml.org>`_ datasets.
"""
import copy
import logging
import os
import re
from typing import Generic, Tuple, TypeVar, Union

import arff
import numpy as np
import pandas as pd
import pandas.api.types as pat
import openml as oml
import scipy as sci

from ..data import Dataset, DatasetType, Datasplit, Feature
from ..utils import as_list, lazy_property, path_from_split, profile, split_path


log = logging.getLogger(__name__)

# hack (only adding a ? to the regexp pattern) to ensure that '?' values remain quoted when we save dataplits in arff format.
arff._RE_QUOTE_CHARS = re.compile(r'[?"\'\\\s%,\000-\031]', re.UNICODE)


class OpenmlLoader:

    def __init__(self, api_key, cache_dir=None):
        oml.config.apikey = api_key
        if cache_dir:
            oml.config.set_cache_directory(cache_dir)

    @profile(logger=log)
    def load(self, task_id=None, dataset_id=None, fold=0):
        if task_id is not None:
            if dataset_id is not None:
                log.warning("Ignoring dataset id {} as a task id {} was already provided.".format(dataset_id, task_id))
            task = oml.tasks.get_task(task_id)
            dataset = task.get_dataset()
            _, nfolds, _ = task.get_split_dimensions()
            if fold >= nfolds:
                raise ValueError("OpenML task {} only accepts `fold` < {}.".format(task_id, nfolds))
        elif dataset_id is not None:
            raise NotImplementedError("OpenML raw datasets are not supported yet, please use an OpenML task instead.")
            dataset = oml.datasets.get_dataset(dataset_id)
            task = AutoTask(dataset)
            if fold > 0:
                raise ValueError("OpenML raw datasets {} only accepts `fold` = 0.".format(task_id))
        else:
            raise ValueError("A task id or a dataset id are required when using OpenML.")
        return OpenmlDataset(task, dataset, fold)


class AutoTask(oml.OpenMLTask):
    """A minimal task implementation providing only the information necessary to get the logic of this current module working."""

    def __init__(self, oml_dataset: oml.OpenMLDataset):
        self._dataset = oml_dataset
        self._nrows = oml_dataset.qualities['NumberOfInstances']
        self.target_name = oml_dataset.default_target_attribute


    def get_train_test_split_indices(self, fold=0):
        # TODO: make auto split 80% train, 20% test (make this configurable, also random vs sequential) and save it to disk
        pass


class OpenmlDataset(Dataset):

    def __init__(self, oml_task: oml.OpenMLTask, oml_dataset: oml.OpenMLDataset, fold=0):
        super().__init__()
        self._oml_task = oml_task
        self._oml_dataset = oml_dataset
        self.fold = fold
        self._train = None
        self._test = None

    @lazy_property
    def type(self):
        def get_type(card):
            if card > 2:
                return DatasetType.multiclass
            elif card == 2:
                return DatasetType.binary
            elif card == 0:
                return DatasetType.regression
            return None

        nclasses = self._oml_dataset.qualities.get('NumberOfClasses', -1)
        if nclasses >= 0:
            return get_type(nclasses)
        else:
            target = next(f for f in self.features if f.is_target)
            return get_type(len(target.values))

    @property
    @profile(logger=log)
    def train(self):
        self._ensure_split_created()
        return self._train

    @property
    @profile(logger=log)
    def test(self):
        self._ensure_split_created()
        return self._test

    @lazy_property
    @profile(logger=log)
    def features(self):
        has_missing_values = lambda f: f.number_missing_values > 0
        is_target = lambda f: f.name == self._oml_task.target_name
        return [Feature(new_idx,
                        f.name,
                        f.data_type,
                        values=f.nominal_values,
                        has_missing_values=has_missing_values(f),
                        is_target=is_target(f)
                        )
                for new_idx, f in enumerate(f for i, f in sorted(self._oml_dataset.features.items())
                                            if f.name not in self._excluded_attributes)
                ]

    @lazy_property
    def target(self):
        return next(f for f in self.features if f.is_target)

    @property
    def _excluded_attributes(self):
        return (self._oml_dataset.ignore_attribute or []) + as_list(self._oml_dataset.row_id_attribute or [])

    def _ensure_split_created(self):
        if self._train is None or self._test is None:
            self._train = OpenmlDatasplit(self)
            self._test = OpenmlDatasplit(self)

    def _load_data(self, fmt):
        splitter = _get_data_splitter_cls(fmt)(self)
        train, test = splitter.split()
        self._train._data[fmt] = train
        self._test._data[fmt] = test

    def _get_split_paths(self, ext=None):
        sp = split_path(self._oml_dataset.data_file)
        train, test = copy.copy(sp), copy.copy(sp)
        train.basename = f"{train.basename}_train_{self.fold}"
        test.basename = f"{test.basename}_test_{self.fold}"
        if ext:
            train.ext = test.ext = ext
        return path_from_split(train), path_from_split(test)


class OpenmlDatasplit(Datasplit):

    def __init__(self, dataset: OpenmlDataset):
        super().__init__(dataset, 'arff')  # TODO: fix format
        self._data = {}

    def data_path(self, format):
        if format not in __supported_file_formats__:
            raise ValueError(f"Dataset {self.dataset._oml_dataset.name} is only available as a file in one of {__supported_file_formats__} formats.")
        return self._get_data(format)

    @lazy_property
    @profile(logger=log)
    def data(self):
        return self._get_data('array')

    def _get_data(self, fmt):
        if fmt not in self._data or (fmt in __supported_file_formats__ and not os.path.isfile(self._data[fmt])):
            self.dataset._load_data(fmt)
        # print("*****\n", self._data[fmt])
        return self._data[fmt]


T = TypeVar('T')
A = Union[np.ndarray, sci.sparse.csr_matrix]
DF = pd.DataFrame


class DataSplitter(Generic[T]):

    def __init__(self, ds: OpenmlDataset):
        self.ds = ds
        self.train_ind, self.test_ind = ds._oml_task.get_train_test_split_indices(self.ds.fold)

    def split(self) -> Tuple[T, T]:
        pass


class ArraySplitter(DataSplitter[A]):
    format = 'array'

    def __init__(self, ds: OpenmlDataset):
        super().__init__(ds)

    def split(self) -> Tuple[A, A]:
        X, *_ = self.ds._oml_dataset.get_data(dataset_format='array')
        return X[self.train_ind, :], X[self.test_ind, :]


class DataFrameSplitter(DataSplitter[DF]):
    format = 'dataframe'

    def __init__(self, ds: OpenmlDataset):
        super().__init__(ds)

    def split(self) -> Tuple[DF, DF]:
        X, *_ = self.ds._oml_dataset.get_data(dataset_format='dataframe')
        return X.iloc[self.train_ind, :], X.iloc[self.test_ind, :]


class ArffSplitter(DataSplitter[str]):
    format = 'arff'

    def __init__(self, ds: OpenmlDataset):
        super().__init__(ds)

    def split(self) -> Tuple[str, str]:
        train_path, test_path = self.ds._get_split_paths(".arff")
        if not os.path.isfile(train_path) or not os.path.isfile(test_path):
            X, *_ = self.ds._oml_dataset.get_data(dataset_format='dataframe')
            train, test = X.iloc[self.train_ind, :], X.iloc[self.test_ind, :]
            name_template = "{name}_{{split}}_{fold}".format(name=self.ds._oml_dataset.name, fold=self.ds.fold)
            self._save_split(train, train_path, name_template.format(split="train"))
            self._save_split(test, test_path, name_template.format(split="test"))
        return train_path, test_path

    @profile(logger=log)
    def _save_split(self, df, path, name):
        log.debug("Saving %s split dataset to %s.", name, path)
        with open(path, 'w') as file:
            description = f"Split dataset file generated by automlbenchmark from OpenML dataset openml.org/d/{self.ds._oml_dataset.dataset_id}"
            attributes = [(c,
                           ('INTEGER' if pat.is_integer_dtype(dt)
                            else 'REAL' if pat.is_float_dtype(dt)
                            else 'NUMERIC' if pat.is_numeric_dtype(dt)
                            else self._get_categorical_values(c) if pat.is_categorical_dtype(dt)
                            else 'STRING'
                           ))
                          for c, dt in zip(df.columns, df.dtypes)]
            arff.dump({
                'description': description,
                'relation': name,
                'attributes': attributes,
                'data': df.values
            }, file)

    def _get_categorical_values(self, col):
        feat = next((f for f in self.ds._oml_dataset.features.values() if f.name == col), None)
        return feat.nominal_values if feat is not None else None


class CsvSplitter(DataSplitter[str]):
    format = 'csv'

    def __init__(self, ds: OpenmlDataset):
        super().__init__(ds)

    def split(self) -> Tuple[str, str]:
        train_path, test_path = self.ds._get_split_paths(".csv")
        if not os.path.isfile(train_path) or not os.path.isfile(test_path):
            X, *_ = self.ds._oml_dataset.get_data(dataset_format='dataframe')
            train, test = X.iloc[self.train_ind, :], X.iloc[self.test_ind, :]
            train.to_csv(train_path, header=True, index=False)
            test.to_csv(test_path, header=True, index=False)
        return train_path, test_path


__data_splitters__ = [ArraySplitter, DataFrameSplitter, ArffSplitter, CsvSplitter]
__supported_file_formats__ = ['arff', 'csv']


def _get_data_splitter_cls(split_format='array'):
    return next(ds for ds in __data_splitters__ if ds.format == split_format)



