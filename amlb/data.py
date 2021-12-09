"""
**data** module provides abstractions for data manipulation:

- **Dataset** represents the entire dataset used by a job:
  providing simple access to subsets like training set, test set,
  and metadata like target feature, and predictors.
- **Datasplit** represents a subset of the dataset,
  providing access to data, either as a file (``path``),
  or as vectors/arrays (``y`` for target, ``X`` for predictors)
  which can also be encoded (``y_enc``, ``X_enc``)
- **Feature** provides metadata for a given feature/column as well as encoding functions.
"""
from abc import ABC, abstractmethod
from enum import Enum, auto
import logging
from typing import List, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .datautils import Encoder
from .utils import clear_cache, lazy_property, profile, repr_def

log = logging.getLogger(__name__)


AM = Union[np.ndarray, sp.spmatrix]
DF = pd.DataFrame


class Feature:

    def __init__(self, index, name, data_type, values=None, has_missing_values=False, is_target=False):
        """
        :param index: index of the feature in the full data frame.
        :param name: name of the feature.
        :param data_type: one of pandas-compatible type ('int', 'float', 'number', 'bool', 'category', 'string', 'object', 'datetime').
        :param values: for categorical features, the sorted list of accepted values.
        :param has_missing_values: True iff the feature has any missing values in the complete dataset.
        :param is_target: True for the target column.
        """
        self.index = index
        self.name = name
        self.data_type = data_type.lower() if data_type is not None else None
        self.values = self.normalize(values).tolist() if values is not None else None
        self.has_missing_values = has_missing_values
        self.is_target = is_target
        # print(self)

    def is_categorical(self, strict=True):
        if strict:
            return self.data_type == 'category'
        else:
            return self.data_type is not None and not self.is_numerical()

    def is_numerical(self):
        return self.data_type in ['int', 'float', 'number']

    @lazy_property
    def label_encoder(self):
        return Encoder('label' if self.values is not None else 'no-op',
                       target=self.is_target,
                       encoded_type=int if self.is_target and not self.is_numerical() else float,
                       missing_values=[None, np.nan, pd.NA],
                       missing_policy='mask' if self.has_missing_values else 'ignore',
                       normalize_fn=self.normalize
                       ).fit(self.values)

    @lazy_property
    def one_hot_encoder(self):
        return Encoder('one-hot' if self.values is not None else 'no-op',
                       target=self.is_target,
                       encoded_type=int if self.is_target and not self.is_numerical() else float,
                       missing_values=[None, np.nan, pd.NA],
                       missing_policy='mask' if self.has_missing_values else 'ignore',
                       normalize_fn=self.normalize
                       ).fit(self.values)

    def normalize(self, arr):
        return np.char.lower(np.char.strip(np.asarray(arr).astype(str)))

    def __repr__(self):
        return repr_def(self)


class AuxData(ABC):
    
    def __init__(self):
        super().__init__()

    @property
    def path(self) -> str:
        pass

    @property
    @abstractmethod
    def data(self) -> DF:
        """
        :return: the auxiliary data as a pandas DataFrame.
        """
        pass

    @profile(logger=log)
    def release(self, properties=None):
        clear_cache(self, properties)


class Datasplit(ABC):

    def __init__(self, dataset, format):
        """
        :param format: the default format of the data file, obtained through the 'path' property.
        """
        super().__init__()
        self.dataset = dataset
        self.format = format

    @property
    def path(self) -> str:
        return self.data_path(self.format)

    @abstractmethod
    def data_path(self, format: str) -> str:
        """
        :param format: the format requested for the data file. Currently supported formats are 'arff', 'csv'.
        :return: the path to the data-split file in the requested format.
        """
        pass

    @property
    @abstractmethod
    def data(self) -> DF:
        """
        :return: all the columns (predictors + target) as a pandas DataFrame.
        """
        pass

    @lazy_property
    @profile(logger=log)
    def X(self) -> DF:
        """
        :return:the predictor columns as a pandas DataFrame.
        """
        predictors_ind = [p.index for p in self.dataset.predictors]
        return self.data.iloc[:, predictors_ind]

    @lazy_property
    @profile(logger=log)
    def y(self) -> DF:
        """
        :return:the target column as a pandas DataFrame: if you need a Series, just call `y.squeeze()`.
        """
        return self.data.iloc[:, [self.dataset.target.index]]

    @lazy_property
    @profile(logger=log)
    def data_enc(self) -> AM:
        data = np.where(self.data.notna(), self.data.values, None)
        encoded_cols = [f.label_encoder.transform(data[:, f.index]) for f in self.dataset.features]
        # optimize mem usage : frameworks use either raw data or encoded ones,
        # so we can clear the cached raw data once they've been encoded
        self.release(['data', 'X', 'y'])
        return np.hstack(tuple(col.reshape(-1, 1) for col in encoded_cols))

    @lazy_property
    @profile(logger=log)
    def X_enc(self) -> AM:
        # TODO: should we use one_hot_encoder here instead?
        # encoded_cols = [p.label_encoder.transform(self.data[:, p.index]) for p in self.dataset.predictors]
        # return np.hstack(tuple(col.reshape(-1, 1) for col in encoded_cols))
        predictors_ind = [p.index for p in self.dataset.predictors]
        return self.data_enc[:, predictors_ind]

    @lazy_property
    @profile(logger=log)
    def y_enc(self) -> AM:
        # return self.dataset.target.label_encoder.transform(self.y)
        return self.data_enc[:, self.dataset.target.index]

    @profile(logger=log)
    def release(self, properties=None):
        clear_cache(self, properties)


class DatasetType(Enum):
    binary = 1
    multiclass = 2
    regression = 3


class Dataset(ABC):

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def type(self) -> DatasetType:
        """
        :return: the problem type for the current dataset.
        """
        pass

    @property
    @abstractmethod
    def train(self) -> Datasplit:
        """
        :return: the data subset used to train the model.
        """
        pass

    @property
    @abstractmethod
    def test(self) -> Datasplit:
        """
        :return: the data subset used to score the model.
        """
        pass

    @property
    @abstractmethod
    def features(self) -> List[Feature]:
        """
        :return: the list of all features available in the current dataset, target included.
        """
        pass

    @property
    def predictors(self) -> List[Feature]:
        """
        :return: the list of all predictor features available in the current dataset
        """
        return [f for f in self.features if f.name != self.target.name]

    @property
    @abstractmethod
    def target(self) -> Feature:
        """
        :return: the target feature for the current dataset.
        """
        pass

    @profile(logger=log)
    def release(self, properties=None):
        """
        Call this to release cached properties and optimize memory once in-memory data are not needed anymore.
        :param properties:
        """
        self.train.release()
        self.test.release()
        clear_cache(self, properties)
