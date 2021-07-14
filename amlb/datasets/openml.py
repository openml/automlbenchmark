"""
**openml** module implements the abstractions defined in **data** module
to expose `OpenML<https://www.openml.org>`_ datasets.
"""
from abc import abstractmethod
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
from ..resources import config as rconfig
from ..utils import as_list, lazy_property, path_from_split, profile, split_path


log = logging.getLogger(__name__)

# hack (only adding a ? to the regexp pattern) to ensure that '?' values remain quoted when we save dataplits in arff format.
arff._RE_QUOTE_CHARS = re.compile(r'[?"\'\\\s%,\000-\031]', re.UNICODE)


class OpenmlLoader:

    def __init__(self, api_key, cache_dir=None):
        oml.config.apikey = api_key
        if cache_dir:
            oml.config.set_cache_directory(cache_dir)

        if oml.config.retry_policy != "robot":
            log.debug("Setting openml retry_policy from '%s' to 'robot'." % oml.config.retry_policy)
            oml.config.set_retry_policy("robot")

    @profile(logger=log)
    def load(self, task_id=None, dataset_id=None, fold=0):
        if task_id is not None:
            if dataset_id is not None:
                log.warning("Ignoring dataset id {} as a task id {} was already provided.".format(dataset_id, task_id))
            task = oml.tasks.get_task(task_id, download_qualities=False)
            dataset = oml.datasets.get_dataset(task.dataset_id, download_qualities=False)
            _, nfolds, _ = task.get_split_dimensions()
            if fold >= nfolds:
                raise ValueError("OpenML task {} only accepts `fold` < {}.".format(task_id, nfolds))
        elif dataset_id is not None:
            raise NotImplementedError("OpenML raw datasets are not supported yet, please use an OpenML task instead.")
        else:
            raise ValueError("A task id or a dataset id are required when using OpenML.")
        return OpenmlDataset(task, dataset, fold)


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

        if hasattr(self._oml_task, "class_labels"):
            return get_type(len(self._oml_task.class_labels))
        return DatasetType.regression

    @property
    def train(self):
        self._ensure_split_created()
        return self._train

    @property
    def test(self):
        self._ensure_split_created()
        return self._test

    @lazy_property
    @profile(logger=log)
    def features(self):
        has_missing_values = lambda f: f.number_missing_values > 0
        is_target = lambda f: f.name == self._oml_task.target_name
        to_feature_type = lambda dt: ('number' if dt == 'numeric'
                                      else 'category' if dt == 'nominal'
                                      else 'string' if dt == 'string'
                                      else 'datetime' if dt == 'date'
                                      else 'object')
        return [Feature(new_idx,
                        f.name,
                        to_feature_type(f.data_type),
                        values=sorted(f.nominal_values) if f.nominal_values else None,
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

    def _load_full_data(self, fmt):
        X, *_ = self._oml_dataset.get_data(dataset_format=fmt)
        if fmt == 'dataframe' and rconfig().openml.infer_dtypes:
            return X.convert_dtypes()
        return X

    def _get_split_paths(self, ext=None):
        sp = split_path(self._oml_dataset.data_file)
        train, test = copy.copy(sp), copy.copy(sp)
        train.basename = f"{train.basename}_train_{self.fold}"
        test.basename = f"{test.basename}_test_{self.fold}"
        if ext:
            train.extension = test.extension = ext
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
    def data(self) -> pd.DataFrame:
        return self._get_data('dataframe')

    @lazy_property
    @profile(logger=log)
    def data_enc(self) -> np.ndarray:
        return self._get_data('array')

    def _get_data(self, fmt):
        if fmt not in self._data or (fmt in __supported_file_formats__ and not os.path.isfile(self._data[fmt])):
            self.dataset._load_data(fmt)
        return self._data[fmt]

    def release(self, properties=None):
        super().release(properties)
        self._data = {}


T = TypeVar('T')
A = Union[np.ndarray, sci.sparse.csr_matrix]
DF = pd.DataFrame


class DataSplitter(Generic[T]):

    def __init__(self, ds: OpenmlDataset):
        self.ds = ds
        self.train_ind, self.test_ind = ds._oml_task.get_train_test_split_indices(self.ds.fold)

    @abstractmethod
    def split(self) -> Tuple[T, T]:
        pass


class ArraySplitter(DataSplitter[A]):
    format = 'array'

    def __init__(self, ds: OpenmlDataset):
        super().__init__(ds)

    @profile(logger=log)
    def split(self) -> Tuple[A, A]:
        X = self.ds._load_full_data('array')
        return X[self.train_ind, :], X[self.test_ind, :]


class DataFrameSplitter(DataSplitter[DF]):
    format = 'dataframe'

    def __init__(self, ds: OpenmlDataset):
        super().__init__(ds)

    @profile(logger=log)
    def split(self) -> Tuple[DF, DF]:
        X = self.ds._load_full_data('dataframe')
        return X.iloc[self.train_ind, :], X.iloc[self.test_ind, :]


class ArffSplitter(DataSplitter[str]):
    format = 'arff'

    def __init__(self, ds: OpenmlDataset):
        super().__init__(ds)

    @profile(logger=log)
    def split(self) -> Tuple[str, str]:
        train_path, test_path = self.ds._get_split_paths(".arff")
        if not os.path.isfile(train_path) or not os.path.isfile(test_path):
            X = self.ds._load_full_data('dataframe')
            train, test = X.iloc[self.train_ind, :], X.iloc[self.test_ind, :]
            name_template = "{name}_{{split}}_{fold}".format(name=self.ds._oml_dataset.name, fold=self.ds.fold)
            self._save_split(train, train_path, name_template.format(split="train"))
            self._save_split(test, test_path, name_template.format(split="test"))
        return train_path, test_path

    def _save_split(self, df, path, name):
        log.debug("Saving %s split dataset to %s.", name, path)
        with open(path, 'w') as file:
            description = f"Split dataset file generated by automlbenchmark from OpenML dataset openml.org/d/{self.ds._oml_dataset.dataset_id}"
            attributes = [(c,
                           ('INTEGER' if pat.is_integer_dtype(dt)
                            else 'REAL' if pat.is_float_dtype(dt)
                           # columns with all values missing will be interpreted as string by default,
                           # but we can use openml meta-data to find out if it should be considered numeric instead.
                            else 'NUMERIC' if pat.is_numeric_dtype(dt) or self._is_numeric(c)
                            else self._get_categorical_values(c) if pat.is_categorical_dtype(dt)
                            else 'STRING'
                           ))
                          for c, dt in zip(df.columns, df.dtypes)]
            arff.dump(dict(
                description=description,
                relation=name,
                attributes=attributes,
                data=df.values
            ), file)

    def _is_numeric(self, col):
        feat = next((f for f in self.ds._oml_dataset.features.values() if f.name == col), None)
        return feat.data_type.lower() == "numeric"

    def _get_categorical_values(self, col):
        feat = next((f for f in self.ds._oml_dataset.features.values() if f.name == col), None)
        if feat is not None:
            # openml-python converts categorical features which look boolean to
            # boolean values, which always write values as 'True' and 'False',
            # so we need to adapt the header accordingly.
            # Not doing so causes an issue in the R packages.
            if set(v.lower() for v in feat.nominal_values) == {"true", "false"}:
                return sorted(v.lower().capitalize() for v in feat.nominal_values)
            return sorted(feat.nominal_values)
        return None


class CsvSplitter(DataSplitter[str]):
    format = 'csv'

    def __init__(self, ds: OpenmlDataset):
        super().__init__(ds)

    @profile(logger=log)
    def split(self) -> Tuple[str, str]:
        train_path, test_path = self.ds._get_split_paths(".csv")
        if not os.path.isfile(train_path) or not os.path.isfile(test_path):
            X = self.ds._load_full_data('dataframe')
            train, test = X.iloc[self.train_ind, :], X.iloc[self.test_ind, :]
            train.to_csv(train_path, header=True, index=False)
            test.to_csv(test_path, header=True, index=False)
        return train_path, test_path


class ParquetSplitter(DataSplitter[str]):
    format = 'parquet'

    def __init__(self, ds: OpenmlDataset):
        super().__init__(ds)

    @profile(logger=log)
    def split(self) -> Tuple[str, str]:
        train_path, test_path = self.ds._get_split_paths(".parquet")
        if not os.path.isfile(train_path) or not os.path.isfile(test_path):
            X = self.ds._load_full_data('dataframe')
            train, test = X.iloc[self.train_ind, :], X.iloc[self.test_ind, :]
            train.to_parquet(train_path)
            test.to_parquet(test_path)
        return train_path, test_path


__data_splitters__ = [ArraySplitter, DataFrameSplitter, ArffSplitter, CsvSplitter, ParquetSplitter]
__supported_file_formats__ = ['arff', 'csv', 'parquet']


def _get_data_splitter_cls(split_format='array'):
    ds_cls = next((ds for ds in __data_splitters__ if ds.format == split_format), None)
    if ds_cls is None:
        supported = [ds.format for ds in __data_splitters__]
        raise ValueError(f"`{split_format}` is not among supported formats: {supported}.")
    return ds_cls

