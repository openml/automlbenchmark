"""
**openml** module implements the abstractions defined in **data** module
to expose `OpenML<https://www.openml.org>`_ datasets.
"""

from __future__ import annotations

import pathlib
from abc import abstractmethod
import copy
import functools
from functools import cached_property
import logging
import os
import re
from typing import Generic, Tuple, TypeVar, Hashable

import arff
import numpy as np
import pandas as pd
import pandas.api.types as pat
import openml as oml
import xmltodict

from ..benchmarks.openml import load_openml_task_and_data
from ..data import AM, DF, Dataset, DatasetType, Datasplit, Feature
from ..datautils import impute_array
from ..resources import config as rconfig, get as rget
from ..utils import (
    as_list,
    path_from_split,
    profile,
    split_path,
    unsparsify,
)


# https://github.com/openml/automlbenchmark/pull/574#issuecomment-1646179921
try:
    set_openml_cache = oml.config.set_cache_directory
except AttributeError:
    set_openml_cache = oml.config.set_root_cache_directory

log = logging.getLogger(__name__)

# hack (only adding a ? to the regexp pattern) to ensure that '?' values remain quoted when we save dataplits in arff format.
arff._RE_QUOTE_CHARS = re.compile(r'[?"\'\\\s%,\000-\031]', re.UNICODE)

# Fix a bug in openml-python<=0.12.2, see https://github.com/openml/automlbenchmark/issues/350
xmltodict.parse = functools.partial(xmltodict.parse, strip_whitespace=False)


class OpenmlLoader:
    def __init__(self, api_key, cache_dir=None):
        oml.config.apikey = api_key
        if cache_dir:
            set_openml_cache(cache_dir)

        if oml.config.retry_policy != "robot":
            log.debug(
                "Setting openml retry_policy from '%s' to 'robot'."
                % oml.config.retry_policy
            )
            oml.config.set_retry_policy("robot")

    @profile(logger=log)
    def load(self, task_id=None, dataset_id=None, fold=0):
        if task_id is not None:
            if dataset_id is not None:
                log.warning(
                    "Ignoring dataset id {} as a task id {} was already provided.".format(
                        dataset_id, task_id
                    )
                )
            task, dataset = load_openml_task_and_data(task_id, with_data=True)
            _, nfolds, _ = task.get_split_dimensions()
            if fold >= nfolds:
                raise ValueError(
                    "OpenML task {} only accepts `fold` < {}.".format(task_id, nfolds)
                )
        elif dataset_id is not None:
            raise NotImplementedError(
                "OpenML raw datasets are not supported yet, please use an OpenML task instead."
            )
        else:
            raise ValueError(
                "A task id or a dataset id are required when using OpenML."
            )
        return OpenmlDataset(task, dataset, fold)


class OpenmlDataset(Dataset):
    def __init__(
        self, oml_task: oml.OpenMLTask, oml_dataset: oml.OpenMLDataset, fold=0
    ):
        super().__init__()
        self._oml_task = oml_task
        self._oml_dataset = oml_dataset
        self.fold = fold
        self._train = None
        self._test = None
        self._nrows: int | None = None

    @property
    def nrows(self) -> int:
        """
        Returns the number of rows in the dataset.
        
        If the row count has not been computed yet, the full dataset is loaded in dataframe
        format, its length is determined, and the result is cached for subsequent calls.
        """
        if self._nrows is None:
            self._nrows = len(self._load_full_data(fmt="dataframe"))
        return self._nrows

    @cached_property
    def type(self):
        """
        Determine the dataset type based on the number of class labels.
        
        If the OpenML task has a 'class_labels' attribute, the type is inferred from its count:
        more than two labels indicate a multiclass task, exactly two indicate a binary task,
        and zero labels indicate a regression task. If there is exactly one label, None is returned.
        When the 'class_labels' attribute is absent, the dataset defaults to regression.
        """
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

    def inference_subsample_files(
        self,
        fmt: str,
        with_labels: bool = False,
        scikit_safe: bool = False,
        keep_empty_features: bool = False,
    ) -> list[Tuple[int, str]]:
        """Generates n subsamples of size k from the test dataset in `fmt` data format.

        We measure the inference time of the models for various batch sizes
        (number of rows). We generate config.inference_time_measurements.repeats
        subsamples for each of the config.inference_time_measurements.batch_sizes.

        These subsamples are stored to file in the `fmt` format (parquet, arff, or csv).
        The function returns a list of tuples of (batch size, file path).

        Iff `with_labels` is true, the target column will be included in the split file.
        Iff `scikit_safe` is true, categorical values are encoded and missing values
        are imputed.
        """
        seed = rget().seed(self.fold)
        batch_sizes = [
            batch_size
            for batch_size in rconfig().inference_time_measurements.batch_sizes
            if not (
                batch_size > self.nrows
                and rconfig().inference_time_measurements.limit_by_dataset_size
            )
        ]
        return [
            (
                n,
                str(
                    self._inference_subsample(
                        fmt=fmt,
                        n=n,
                        seed=seed + i,
                        with_labels=with_labels,
                        scikit_safe=scikit_safe,
                        keep_empty_features=keep_empty_features,
                    )
                ),
            )
            for n in batch_sizes
            for i, _ in enumerate(range(rconfig().inference_time_measurements.repeats))
        ]

    @profile(logger=log)
    def _inference_subsample(
        self,
        fmt: str,
        n: int,
        seed: int = 0,
        with_labels: bool = False,
        scikit_safe: bool = False,
        keep_empty_features: bool = False,
    ) -> pathlib.Path:
        """
        Generates and writes an inference subsample of the test split to disk.
        
        Extracts a random subsample of n rows from the test split and saves it as a file
        in the specified format. If with_labels is True, the target column is included;
        if scikit_safe is True, categorical features and missing values are processed
        via encoding and imputation. The keep_empty_features flag controls whether columns
        with all missing values are retained (with zeros imputed) or removed.
        
        The subsample file is named based on the original test split file, the number of
        samples, and the seed value.
        
        Args:
            fmt: The file format for the output file ('csv', 'arff', or 'parquet').
            n: The number of samples to include in the subsample.
            seed: Random seed for reproducibility (default 0).
            with_labels: If True, includes the target column in the subsample.
            scikit_safe: If True, applies encoding and imputation to ensure scikit-learn
                compatibility.
            keep_empty_features: If True, retains columns that contain only missing values,
                imputing them as 0; otherwise, such columns are removed.
        
        Returns:
            The file path of the saved subsample file.
        
        Raises:
            ValueError: If fmt is not one of 'csv', 'arff', or 'parquet'.
        """

        def get_non_empty_columns(data: DF) -> list[Hashable]:
            return [
                c for c, is_empty in data.isnull().all(axis=0).items() if not is_empty
            ]

        # Just a hack for now, the splitters all work specifically with openml tasks.
        # The important thing is that we split to disk and can load it later.

        # We should consider taking a stratified sample if n is large enough,
        # inference time might differ based on class
        if scikit_safe:
            if with_labels:
                _, data = impute_array(
                    self.train.data_enc,
                    self.test.data_enc,
                    keep_empty_features=keep_empty_features,
                )
                columns = (
                    self.train.data.columns
                    if keep_empty_features
                    else get_non_empty_columns(self.train.data)
                )
            else:
                _, data = impute_array(
                    self.train.X_enc,
                    self.test.X_enc,
                    keep_empty_features=keep_empty_features,
                )
                columns = (
                    self.train.X.columns
                    if keep_empty_features
                    else get_non_empty_columns(self.train.X)
                )

            # `impute_array` drops columns that only have missing values
            data = pd.DataFrame(data, columns=columns)
        else:
            data = self._test.data if with_labels else self._test.X

        subsample = data.sample(
            n=n,
            replace=True,
            random_state=seed,
        )

        _, test_path = self._get_split_paths()
        test_path = pathlib.Path(test_path)
        subsample_path = test_path.parent / f"{test_path.stem}_{n}_{seed}.{fmt}"
        if fmt == "csv":
            subsample.to_csv(subsample_path, header=True, index=False)
        elif fmt == "arff":
            ArffSplitter(self)._save_split(
                subsample,
                subsample_path,
                name=f"{self._oml_dataset.name}_inference_{self.fold}_{n}",
            )
        elif fmt == "parquet":
            subsample.to_parquet(subsample_path)
        else:
            msg = f"{fmt=}, but must be one of 'csv', 'arff', or 'parquet'."
            raise ValueError(msg)

        return subsample_path

    @cached_property
    @profile(logger=log)
    def features(self):
        """
        Returns a list of Feature objects representing the dataset's features.
        
        Iterates over the dataset metadata, excluding attributes listed in the exclusion set,
        and constructs a Feature for each valid attribute with an index, mapped type, sorted nominal
        values (if applicable), and flags indicating missing values and target status.
        """
        def has_missing_values(f) -> bool:
            return f.number_missing_values > 0

        def is_target(f) -> bool:
            return f.name == self._oml_task.target_name

        def to_feature_type(dt):
            if dt == "numeric":
                return "number"
            if dt == "nominal":
                return "category"
            if dt == "string":
                return "string"
            if dt == "date":
                return "datetime"
            return "object"

        return [
            Feature(
                new_idx,
                f.name,
                to_feature_type(f.data_type),
                values=sorted(f.nominal_values) if f.nominal_values else None,
                has_missing_values=has_missing_values(f),
                is_target=is_target(f),
            )
            for new_idx, f in enumerate(
                f
                for i, f in sorted(self._oml_dataset.features.items())
                if f.name not in self._excluded_attributes
            )
        ]

    @cached_property
    def target(self):
        """
        Returns the target feature of the dataset.
        
        Searches through the list of features and returns the first one flagged as the target.
        Raises StopIteration if no feature is marked as the target.
        """
        return next(f for f in self.features if f.is_target)

    @property
    def _excluded_attributes(self):
        return (self._oml_dataset.ignore_attribute or []) + as_list(
            self._oml_dataset.row_id_attribute or []
        )

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
        if fmt == "dataframe" and rconfig().openml.infer_dtypes:
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
        super().__init__(dataset, "arff")
        self._data: dict[str, AM | DF | str] = {}

    def data_path(self, format):
        """
        Return the file path for the dataset in the specified format.
        
        Args:
            format (str): The desired file format. Must be one of the supported formats defined by __supported_file_formats__.
        
        Raises:
            ValueError: If the format is not among the supported file formats.
        
        Returns:
            The file path corresponding to the dataset in the given format.
        """
        if format not in __supported_file_formats__:
            raise ValueError(
                f"Dataset {self.dataset._oml_dataset.name} is only available as a file in one of {__supported_file_formats__} formats."
            )
        return self._get_data(format)

    @cached_property
    @profile(logger=log)
    def data(self) -> DF:
        """
        Retrieves the dataset split as a DataFrame.
        
        This method returns the split data in a Pandas DataFrame format by invoking the internal
        data retrieval helper.
        """
        return self._get_data("dataframe")

    @cached_property
    @profile(logger=log)
    def data_enc(self) -> AM:
        """
        Return the dataset in encoded array format.
        
        Retrieves the dataset as an encoded array by invoking the generic data loader with the
        "array" option.
        
        Returns:
            AM: The encoded dataset as an array.
        """
        return self._get_data("array")

    def _get_data(self, fmt):
        if fmt not in self._data or (
            fmt in __supported_file_formats__ and not os.path.isfile(self._data[fmt])
        ):
            self.dataset._load_data(fmt)
        return self._data[fmt]

    def release(self, properties=None):
        super().release(properties)
        self._data = {}


T = TypeVar("T")


class DataSplitter(Generic[T]):
    def __init__(self, ds: OpenmlDataset):
        self.ds = ds
        self.train_ind, self.test_ind = ds._oml_task.get_train_test_split_indices(
            self.ds.fold
        )

    @abstractmethod
    def split(self) -> Tuple[T, T]:
        pass


class ArraySplitter(DataSplitter[AM]):
    format = "array"

    def __init__(self, ds: OpenmlDataset):
        super().__init__(ds)

    @profile(logger=log)
    def split(self) -> Tuple[AM, AM]:
        X = self.ds._load_full_data("array")
        return X[self.train_ind, :], X[self.test_ind, :]


class DataFrameSplitter(DataSplitter[DF]):
    format = "dataframe"

    def __init__(self, ds: OpenmlDataset):
        super().__init__(ds)

    @profile(logger=log)
    def split(self) -> Tuple[DF, DF]:
        X = self.ds._load_full_data("dataframe")
        return X.iloc[self.train_ind, :], X.iloc[self.test_ind, :]


class ArffSplitter(DataSplitter[str]):
    format = "arff"

    def __init__(self, ds: OpenmlDataset):
        super().__init__(ds)

    @profile(logger=log)
    def split(self) -> Tuple[str, str]:
        train_path, test_path = self.ds._get_split_paths(".arff")
        if not os.path.isfile(train_path) or not os.path.isfile(test_path):
            X = self.ds._load_full_data("dataframe")
            train, test = X.iloc[self.train_ind, :], X.iloc[self.test_ind, :]
            name_template = "{name}_{{split}}_{fold}".format(
                name=self.ds._oml_dataset.name, fold=self.ds.fold
            )
            self._save_split(train, train_path, name_template.format(split="train"))
            self._save_split(test, test_path, name_template.format(split="test"))
        return train_path, test_path

    def _save_split(self, df, path, name):
        log.debug("Saving %s split dataset to %s.", name, path)
        with open(path, "w") as file:
            description = f"Split dataset file generated by automlbenchmark from OpenML dataset openml.org/d/{self.ds._oml_dataset.dataset_id}"

            def determine_arff_type(
                column_name: str, dtype: np.dtype | pd.core.dtypes.base.ExtensionDtype
            ) -> str | list[str]:
                if pat.is_integer_dtype(dtype):
                    return "INTEGER"
                if pat.is_float_dtype(dtype):
                    return "REAL"
                if pat.is_bool_dtype(dtype) and not self._is_numeric(column_name):
                    # We observe OpenML annotation on determining how to treat a bool
                    # Bools will match on `is_numeric_dtype` as well.
                    return self._get_categorical_values(column_name)
                if pat.is_numeric_dtype(dtype):
                    return "NUMERIC"
                if pat.is_categorical_dtype(dtype):
                    return self._get_categorical_values(column_name)
                return "STRING"

            attributes = [
                (c, determine_arff_type(c, dt)) for c, dt in zip(df.columns, df.dtypes)
            ]
            arff.dump(
                dict(
                    description=description,
                    relation=name,
                    attributes=attributes,
                    data=df.values,
                ),
                file,
            )

    def _is_numeric(self, col):
        feat = next(
            (f for f in self.ds._oml_dataset.features.values() if f.name == col), None
        )
        return feat.data_type.lower() == "numeric"

    def _get_categorical_values(self, col):
        feat = next(
            (f for f in self.ds._oml_dataset.features.values() if f.name == col), None
        )
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
    format = "csv"

    def __init__(self, ds: OpenmlDataset):
        super().__init__(ds)

    @profile(logger=log)
    def split(self) -> Tuple[str, str]:
        train_path, test_path = self.ds._get_split_paths(".csv")
        if not os.path.isfile(train_path) or not os.path.isfile(test_path):
            X = self.ds._load_full_data("dataframe")
            train, test = X.iloc[self.train_ind, :], X.iloc[self.test_ind, :]
            train.to_csv(train_path, header=True, index=False)
            test.to_csv(test_path, header=True, index=False)
        return train_path, test_path


class ParquetSplitter(DataSplitter[str]):
    format = "parquet"

    def __init__(self, ds: OpenmlDataset):
        super().__init__(ds)

    @profile(logger=log)
    def split(self) -> Tuple[str, str]:
        train_path, test_path = self.ds._get_split_paths(".parquet")
        if not os.path.isfile(train_path) or not os.path.isfile(test_path):
            X = self.ds._load_full_data("dataframe")
            # arrow (used to write parquet files) doesn't support sparse dataframes: https://github.com/apache/arrow/issues/1894
            train, test = unsparsify(
                X.iloc[self.train_ind, :], X.iloc[self.test_ind, :]
            )
            train.to_parquet(train_path)
            test.to_parquet(test_path)
        return train_path, test_path


__data_splitters__ = [
    ArraySplitter,
    DataFrameSplitter,
    ArffSplitter,
    CsvSplitter,
    ParquetSplitter,
]
__supported_file_formats__ = ["arff", "csv", "parquet"]


def _get_data_splitter_cls(split_format="array"):
    ds_cls = next((ds for ds in __data_splitters__ if ds.format == split_format), None)
    if ds_cls is None:
        supported = [ds.format for ds in __data_splitters__]
        raise ValueError(
            f"`{split_format}` is not among supported formats: {supported}."
        )
    return ds_cls
