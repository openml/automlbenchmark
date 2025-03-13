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

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from functools import cached_property
import logging
from typing import List, Union, Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing_extensions import TypeAlias

from .datautils import Encoder
from .utils import clear_cache, profile, repr_def

log = logging.getLogger(__name__)


AM: TypeAlias = Union[np.ndarray, sp.spmatrix]
DF: TypeAlias = pd.DataFrame


class Feature:
    def __init__(
        self,
        index: int,
        name: str,
        data_type: str | None,
        values: Iterable[str] | None = None,
        has_missing_values: bool = False,
        is_target: bool = False,
    ):
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
        self.values = values  # type: ignore  # https://github.com/python/mypy/issues/3004
        self.has_missing_values = has_missing_values
        self.is_target = is_target

    def is_categorical(self, strict: bool = True) -> bool:
        if strict:
            return self.data_type == "category"
        return self.data_type is not None and not self.is_numerical()

    def is_numerical(self) -> bool:
        """
        Determines if the feature is numerical.
        
        Checks whether the feature's data_type attribute is one of "int", "float", or "number"
        to indicate that it represents numerical data.
        
        Returns:
            bool: True if the feature is numerical, False otherwise.
        """
        return self.data_type in ["int", "float", "number"]

    @cached_property
    def label_encoder(self) -> Encoder:
        """
        Creates and returns a fitted encoder for the feature's values.
        
        The encoder is configured based on the feature's properties. It uses "label" encoding if there are values present (or "no-op" otherwise), sets the encoded type to int for non-numerical target features (and float otherwise), and applies masking for missing values if they exist. The encoder also uses the feature's normalization function and is fitted on the current values.
        """
        return Encoder(
            "label" if self.values is not None else "no-op",
            target=self.is_target,
            encoded_type=int if self.is_target and not self.is_numerical() else float,
            missing_values=[None, np.nan, pd.NA],
            missing_policy="mask" if self.has_missing_values else "ignore",
            normalize_fn=Feature.normalize,
        ).fit(self.values)

    @cached_property
    def one_hot_encoder(self) -> Encoder:
        """
        Creates and fits a one-hot encoder for the feature.
        
        The encoder is instantiated to perform one-hot encoding if the feature has defined
        values, or to act as a no-op encoder otherwise. It is configured based on whether the
        feature is a target (using an integer type for non-numerical targets and float otherwise),
        handles missing values by masking when necessary, and applies a normalization function.
        The encoder is then fit on the feature's values before being returned.
        """
        return Encoder(
            "one-hot" if self.values is not None else "no-op",
            target=self.is_target,
            encoded_type=int if self.is_target and not self.is_numerical() else float,
            missing_values=[None, np.nan, pd.NA],
            missing_policy="mask" if self.has_missing_values else "ignore",
            normalize_fn=Feature.normalize,
        ).fit(self.values)

    @staticmethod
    def normalize(arr: Iterable[str]) -> np.ndarray:
        return np.char.lower(np.char.strip(np.asarray(arr).astype(str)))

    @property
    def values(self) -> list[str] | None:
        return self._values

    @values.setter
    def values(self, values: Iterable[str]) -> None:
        self._values = (
            Feature.normalize(values).tolist() if values is not None else None
        )

    def __repr__(self) -> str:
        return repr_def(self, "all")


class Datasplit(ABC):
    def __init__(self, dataset: Dataset, file_format: str):
        """
        :param file_format: the default format of the data file, obtained through the 'path' property.
        """
        super().__init__()
        self.dataset = dataset
        self.format = file_format

    @property
    def path(self) -> str:
        return self.data_path(self.format)

    @abstractmethod
    def data_path(self, format: str) -> str:
        """
        Return the file path for the data split in the specified format.
        
        The format parameter determines which data file path is returned. Supported formats include 'arff' and 'csv'.
        
        Args:
            format (str): The requested data file format.
        
        Returns:
            str: The path to the data-split file in the requested format.
        """
        pass

    @cached_property
    @abstractmethod
    def data(self) -> DF:
        """
        Returns all columns of the data split as a pandas DataFrame.
        
        This includes both predictor features and the target variable.
        """
        pass

    @cached_property
    @profile(logger=log)
    def X(self) -> DF:
        """
        Return the predictor columns as a pandas DataFrame.
        
        This property extracts and returns the subset of columns corresponding to the
        predictor features defined in the parent dataset.
        """
        predictors_ind = [p.index for p in self.dataset.predictors]
        return self.data.iloc[:, predictors_ind]

    @cached_property
    @profile(logger=log)
    def y(self) -> DF:
        """
        Return the target column as a pandas DataFrame.
        
        Extracts the target column from the underlying data using the feature index defined by the parent dataset.
        If a pandas Series is preferred, apply the squeeze() method to the returned DataFrame.
        """
        return self.data.iloc[:, [self.dataset.target.index]]  # type: ignore

    @cached_property
    @profile(logger=log)
    def data_enc(self) -> AM:
        """
        Encodes dataset features into a 2D NumPy array.
        
        Transforms each feature column by applying its label encoder to the corresponding
        raw data column, reshapes the output into a column vector, and concatenates all
        encoded columns horizontally. Cached raw data properties ('data', 'X', and 'y')
        are released to optimize memory usage.
        """
        encoded_cols = [
            f.label_encoder.transform(self.data.iloc[:, f.index])
            for f in self.dataset.features
        ]
        # optimize mem usage : frameworks use either raw data or encoded ones,
        # so we can clear the cached raw data once they've been encoded
        self.release(["data", "X", "y"])
        return np.hstack(tuple(col.reshape(-1, 1) for col in encoded_cols))  # type: ignore[union-attr]

    @cached_property
    @profile(logger=log)
    def X_enc(self) -> AM:
        """
        Returns the encoded predictor features.
        
        Extracts and returns the subset of the encoded data corresponding to the predictor columns,
        using the indices of the predictors defined in the parent dataset.
        """
        predictors_ind = [p.index for p in self.dataset.predictors]
        return self.data_enc[:, predictors_ind]

    @cached_property
    @profile(logger=log)
    def y_enc(self) -> AM:
        # return self.dataset.target.label_encoder.transform(self.y)
        """
        Returns the encoded target column.
        
        Extracts and returns the encoded target values from the precomputed data array by selecting the
        column corresponding to the target feature's index.
        
        Returns:
            AM: An array or sparse matrix containing the encoded target values.
        """
        return self.data_enc[:, self.dataset.target.index]

    @profile(logger=log)
    def release(self, properties: Iterable[str] | None = None) -> None:
        clear_cache(self, properties)


class DatasetType(Enum):
    binary = 1
    multiclass = 2
    regression = 3
    timeseries = 4


class Dataset(ABC):
    def __init__(self) -> None:
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
    def release(self) -> None:
        """
        Call this to release cached properties and optimize memory once in-memory data are not needed anymore.
        """
        self.train.release()
        self.test.release()
        clear_cache(self)
