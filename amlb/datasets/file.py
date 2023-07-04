from abc import abstractmethod
import logging
import math
import os
import re
import tempfile
from typing import List, Union

import arff
import numpy as np
import pandas as pd
import pandas.api.types as pat

from ..data import Dataset, DatasetType, Datasplit, Feature
from ..datautils import read_csv, to_data_frame
from ..resources import config as rconfig
from ..utils import Namespace as ns, as_list, lazy_property, list_all_files, memoize, path_from_split, profile, repr_def, split_path

from .fileutils import is_archive, is_valid_url, unarchive_file, get_file_handler
from copy import deepcopy

log = logging.getLogger(__name__)

train_search_pat = re.compile(r"(?:(.*)[_-])train(?:[_-](\d+))?\.\w+")
test_search_pat = re.compile(r"(?:(.*)[_-])test(?:[_-](\d+))?\.\w+")


class FileLoader:

    def __init__(self, cache_dir=None):
        self._cache_dir = cache_dir if cache_dir else tempfile.mkdtemp(prefix='amlb_cache')

    @profile(logger=log)
    def load(self, dataset, fold=0):
        dataset = dataset if isinstance(dataset, ns) else ns(path=dataset)
        log.debug("Loading dataset %s", dataset)
        paths = self._extract_train_test_paths(dataset.path if 'path' in dataset else dataset, fold=fold, name=dataset['name'] if 'name' in dataset else None)
        assert fold < len(paths['train']), f"No training dataset available for fold {fold} among dataset files {paths['train']}"
        assert fold < len(paths['test']), f"No test dataset available for fold {fold} among dataset files {paths['test']}"

        target = dataset['target']
        type_ = dataset['type']
        features = dataset['features']
        ext = os.path.splitext(paths['train'][fold])[1].lower()
        train_path = paths['train'][fold]
        test_path = paths['test'][fold] if len(paths['test']) > 0 else None
        log.info(f"Using training set {train_path} with test set {test_path}.")
        if ext == '.arff':
            return ArffDataset(train_path, test_path, target=target, features=features, type=type_)
        elif ext == '.csv':
            if type_ is not None and DatasetType[type_] == DatasetType.timeseries and dataset['timestamp_column'] is None:
                log.warning("Warning: For timeseries task setting undefined timestamp column to `timestamp`.")
                dataset = deepcopy(dataset)
                dataset['timestamp_column'] = "timestamp"
            csv_dataset = CsvDataset(train_path, test_path, target=target, features=features, type=type_, timestamp_column=dataset['timestamp_column'] if 'timestamp_column' in dataset else None)
            if csv_dataset.type == DatasetType.timeseries:
                csv_dataset = self.extend_dataset_with_timeseries_config(csv_dataset, dataset)
            return csv_dataset
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _extract_train_test_paths(self, dataset, fold=None, name=None):
        if isinstance(dataset, (tuple, list)):
            assert len(dataset) % 2 == 0, "dataset list must contain an even number of paths: [train_0, test_0, train_1, test_1, ...]."
            return self._extract_train_test_paths(ns(train=[p for i, p in enumerate(dataset) if i % 2 == 0],
                                                     test=[p for i, p in enumerate(dataset) if i % 2 == 1]),
                                                  fold=fold, name=name)
        elif isinstance(dataset, ns):
            return dict(train=[self._extract_train_test_paths(p, name=name)['train'][0]
                               if i == fold else None
                               for i, p in enumerate(as_list(dataset.train))],
                        test=[self._extract_train_test_paths(p, name=name)['train'][0]
                              if i == fold else None
                              for i, p in enumerate(as_list(dataset.test))])
        else:
            assert isinstance(dataset, str)
            dataset = os.path.expanduser(dataset)
            dataset = dataset.format(**rconfig().common_dirs)

        if os.path.exists(dataset):
            if os.path.isfile(dataset):
                if is_archive(dataset):
                    arch_name, _ = os.path.splitext(os.path.basename(dataset))
                    dest_folder = os.path.join(self._cache_dir, arch_name)
                    if not os.path.exists(dest_folder):  # don't uncompress if previously done
                        dest_folder = unarchive_file(dataset, dest_folder)
                    return self._extract_train_test_paths(dest_folder)
                else:
                    return dict(train=[dataset], test=[])
            elif os.path.isdir(dataset):
                files = list_all_files(dataset)
                log.debug("Files found in dataset folder %s: %s", dataset, files)
                assert len(files) > 0, f"Empty folder: {dataset}"
                if len(files) == 1:
                    return dict(train=files, test=[])

                train_matches = [m for m in [train_search_pat.search(f) for f in files] if m]
                test_matches = [m for m in [test_search_pat.search(f) for f in files] if m]
                # verify they're for the same dataset (just based on name)
                assert train_matches and test_matches, f"Folder {dataset} must contain at least one training and one test dataset."
                root_names = {m[1] for m in (train_matches+test_matches)}
                assert len(root_names) == 1, f"All dataset files in {dataset} should follow the same naming: xxxxx_train_N.ext or xxxxx_test_N.ext with N starting from 0."

                train_no_fold = next((m[0] for m in train_matches if m[2] is None), None)
                test_no_fold = next((m[0] for m in test_matches if m[2] is None), None)
                if train_no_fold and test_no_fold:
                    return dict(train=[train_no_fold], test=[test_no_fold])

                paths = dict(train=[], test=[])
                fold = 0
                while fold >= 0:
                    train = next((m[0] for m in train_matches if m[2] == str(fold)), None)
                    test = next((m[0] for m in test_matches if m[2] == str(fold)), None)
                    if train and test:
                        paths['train'].append(train)
                        paths['test'].append(test)
                        fold += 1
                    else:
                        fold = -1
                assert len(paths) > 0, f"No dataset file found in {dataset}: they should follow the naming xxxx_train.ext, xxxx_test.ext or xxxx_train_0.ext, xxxx_test_0.ext, xxxx_train_1.ext, ..."
                return paths
        elif is_valid_url(dataset):
            if name is None:
                cached_file = os.path.join(self._cache_dir, os.path.basename(dataset))
            else:
                cached_file = os.path.join(self._cache_dir, name, os.path.basename(dataset))
            if not os.path.exists(cached_file):  # don't download if previously done
                handler = get_file_handler(dataset)
                assert handler.exists(dataset), f"Invalid path/url: {dataset}"
                handler.download(dataset, dest_path=cached_file)
            return self._extract_train_test_paths(cached_file)
        else:
            raise ValueError(f"Invalid dataset description: {dataset}")

    def __repr__(self):
        return repr_def(self)

    def extend_dataset_with_timeseries_config(self, dataset, dataset_config):
        dataset = deepcopy(dataset)
        dataset_config = deepcopy(dataset_config)
        if dataset_config['forecast_horizon_in_steps'] is None:
            raise AssertionError("Task definition for timeseries must include `forecast_horizon_in_steps`")
        if dataset_config['seasonality'] is None:
            raise AssertionError("Task definition for timeseries must include `seasonality`")
        if dataset_config['id_column'] is None:
            log.warning("Warning: For timeseries task, setting undefined `id_column` to `item_id`.")
            dataset_config['id_column'] = 'item_id'

        dataset.timestamp_column = dataset_config['timestamp_column']
        dataset.id_column = dataset_config['id_column']
        dataset.forecast_horizon_in_steps = int(dataset_config['forecast_horizon_in_steps'])
        dataset.seasonality = int(dataset_config['seasonality'])

        num_timeseries = dataset.train.data[dataset.id_column].nunique()
        if (len(dataset.test.data) - len(dataset.train.data)) / num_timeseries != dataset.forecast_horizon_in_steps:
            raise AssertionError("Test dataset must contain train data + values in the forecast horizon")

        # Compute in-sample seasonal error - needed later for metrics like MASE.
        # We need to store this information here because Result object has no access to past time series values.
        train_set_with_index = dataset.train.data.set_index(dataset.id_column)
        seasonal_diffs = train_set_with_index.groupby(level=dataset.id_column, sort=False).diff(dataset.seasonality).abs()
        abs_seasonal_error = seasonal_diffs.groupby(level=dataset.id_column, sort=False).mean().fillna(1.0).values
        # Repeat seasonal error & item_id for each time step in the forecast horizon
        dataset.repeated_abs_seasonal_error = np.repeat(abs_seasonal_error, dataset.forecast_horizon_in_steps)
        dataset.repeated_item_id = np.repeat(np.arange(num_timeseries), dataset.forecast_horizon_in_steps)

        min_context_length_data_train = dataset.train.X.groupby(dataset.id_column).count().values.min()
        min_sequence_length_data_test = dataset.test.X.groupby(dataset.id_column).count().values.min()
        min_context_length_data_test = min_sequence_length_data_test - dataset.forecast_horizon_in_steps
        if min_context_length_data_test < 1:
            raise ValueError(f'The minimum context length of the test data is smaller than 1. Consider reducing the forecast horizon to {min_sequence_length_data_test - 1}.')
        if min_context_length_data_train < 2 * dataset.forecast_horizon_in_steps + 1:
            log.warning(f'The minimum context length of the train data is smaller than (2 x forecast horizon + 1). Consider reducing the forecast horizon to {math.floor((min_context_length_data_train - 1) / 2)}.')
        if min_context_length_data_test < 2 * dataset.forecast_horizon_in_steps:
            log.warning(f'The minimum context length of the train data is smaller than (2 x forecast horizon). Consider reducing the forecast horizon to {math.floor((min_context_length_data_train) / 2)}.')

        return dataset



class FileDataset(Dataset):

    def __init__(self, train: Datasplit, test: Datasplit,
                 target: Union[int, str] = None, features: List[Union[ns, str]] = None, type: str = None):
        super().__init__()
        self._train = train
        self._test = test
        self._target = target
        self._features = features
        self._type = type

    @property
    def type(self) -> DatasetType:
        assert self.target is not None
        return (DatasetType[self._type] if self._type is not None
                else DatasetType.regression if self.target.values is None
                else DatasetType.binary if len(self.target.values) == 2
                else DatasetType.multiclass)

    @property
    def train(self) -> Datasplit:
        return self._train

    @property
    def test(self) -> Datasplit:
        return self._test

    @property
    def features(self) -> List[Feature]:
        return self._get_metadata('features')

    @property
    def target(self) -> Feature:
        return self._get_metadata('target')

    @memoize
    def _get_metadata(self, prop):
        meta = self._train.load_metadata()
        return meta[prop]

    def __repr__(self):
        return repr_def(self, 'all')


class FileDatasplit(Datasplit):

    def __init__(self, dataset: FileDataset, format: str, path: str):
        super().__init__(dataset, format)
        self._path = path
        self._data = {format: path}

    def data_path(self, format):
        supported_formats = [cls.format for cls in __file_converters__]
        if format not in supported_formats:
            name = split_path(self._path).basename
            raise ValueError(f"Dataset {name} is only available in one of {supported_formats} formats.")
        return self._get_data(format)

    @lazy_property
    def data(self):
        # use codecs for unicode support: path = codecs.load(self._path, 'rb', 'utf-8')
        log.debug("Loading datasplit %s.", self.path)
        return self.load_data()

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def load_metadata(self):
        pass

    def _get_data(self, fmt):
        if fmt not in self._data:
            converter = _get_file_convert_cls(fmt)()
            self._data[fmt] = converter.convert(self)
        return self._data[fmt]

    def _find_target_feature(self, features: List[Feature]):
        target = self.dataset._target
        return (features[target] if isinstance(target, int)
                else next(f for f in features if f.name == target) if isinstance(target, str)
                else next((f for f in features if f.name.lower() in ['target', 'class']), None) or features[-1])

    def _set_feature_as_target(self, target: Feature):
        # for classification problems, ensure that the target appears as categorical
        ds_type = self.dataset._type
        if ds_type and DatasetType[ds_type] in [DatasetType.binary, DatasetType.multiclass]:
            if not target.is_categorical():
                log.warning("Forcing target column `%s` as 'category' for classification problems: was originally detected as '%s'.",
                            target.name, target.data_type)
                self._convert_to_categorical(target)
        target.is_target = True

    def _convert_to_categorical(self, feature: Feature):
        feature.data_type = 'category'

    def __repr__(self):
        return repr_def(self, 'all')


class ArffDataset(FileDataset):

    def __init__(self, train_path, test_path,
                 target=None, features=None, type=None):
        # todo: handle auto-split (if test_path is None): requires loading the training set, split, save
        super().__init__(ArffDatasplit(self, train_path), ArffDatasplit(self, test_path),
                         target=target, features=features, type=type)


class ArffDatasplit(FileDatasplit):

    def __init__(self, dataset, path):
        super().__init__(dataset, format='arff', path=path)
        self._ds = None

    def _ensure_loaded(self):
        if self._ds is None:
            with open(self.path) as f:
                self._ds = arff.load(f)

    @profile(logger=log)
    def load_metadata(self):
        self._ensure_loaded()
        attrs = self._ds['attributes']
        # arff loader types = ['NUMERIC', 'REAL', 'INTEGER', 'STRING']
        to_feature_type = lambda arff_type: ('category' if isinstance(arff_type, (list, set))
                                             else 'string' if arff_type.lower() == 'string'
                                             else 'int' if arff_type.lower() == 'integer'
                                             else 'float' if arff_type.lower() == 'real'
                                             else 'number' if arff_type.lower() == 'numeric'
                                             else 'object')
        features = [Feature(i, attr[0], to_feature_type(attr[1]))
                    for i, attr in enumerate(attrs)]
        target = self._find_target_feature(features)
        self._set_feature_as_target(target)

        df = to_data_frame(self._ds['data'])
        for f in features:
            col = df.iloc[:, f.index]
            f.has_missing_values = col.hasnans
            if f.is_categorical():
                arff_type = attrs[f.index][1]
                assert isinstance(arff_type, (list, set))
                f.values = sorted(arff_type)

        meta = dict(
            features=features,
            target=target
        )
        log.debug("Metadata for dataset %s: %s", self.path, meta)
        return meta

    @profile(logger=log)
    def load_data(self):
        self._ensure_loaded()
        columns = [f.name for f in self.dataset.features]
        df = pd.DataFrame(self._ds['data'], columns=columns)
        dt_conversions = {f.name: f.data_type
                          for f in self.dataset.features
                          if f.data_type == 'category'}
        if dt_conversions:
            df = df.astype(dt_conversions, copy=False)
        return df

    def release(self, properties=None):
        super().release(properties)
        self._ds = None


class CsvDataset(FileDataset):

    def __init__(self, train_path, test_path,
                 target=None, features=None, type=None, timestamp_column=None):
        # todo: handle auto-split (if test_path is None): requires loading the training set, split, save
        super().__init__(None, None,
                         target=target, features=features, type=type)
        self._train = CsvDatasplit(self, train_path, timestamp_column=timestamp_column)
        self._test = CsvDatasplit(self, test_path, timestamp_column=timestamp_column)
        self._dtypes = None


class CsvDatasplit(FileDatasplit):

    def __init__(self, dataset, path, timestamp_column=None):
        super().__init__(dataset, format='csv', path=path)
        self._ds = None
        self.timestamp_column = timestamp_column

    def _ensure_loaded(self):
        if self._ds is None:
            if self.dataset._dtypes is None:
                df = read_csv(self.path, timestamp_column=self.timestamp_column)
                # df = df.convert_dtypes()
                dt_conversions = {name: 'category'
                                  for name, dtype in zip(df.dtypes.index, df.dtypes.values)
                                  if pat.is_string_dtype(dtype)
                                  or pat.is_object_dtype(dtype)
                                  or (name == self.dataset._target
                                      and self.dataset._type is not None
                                      and DatasetType[self.dataset._type] in [DatasetType.binary, DatasetType.multiclass])
                                  }
                # we could be a bit more clever in the future and convert 'string' to category iff len(distinct values) << nrows
                if dt_conversions:
                    df = df.astype(dt_conversions, copy=False)

                self._ds = df
                self.dataset._dtypes = self._ds.dtypes

            else:
                self._ds = read_csv(self.path, dtype=self.dataset._dtypes.to_dict(), timestamp_column=self.timestamp_column)

    @profile(logger=log)
    def load_metadata(self):
        self._ensure_loaded()
        dtypes = self.dataset._dtypes
        to_feature_type = lambda dt: ('int' if pat.is_integer_dtype(dt)
                                      else 'float' if pat.is_float_dtype(dt)
                                      else 'number' if pat.is_numeric_dtype(dt)
                                      else 'category' if pat.is_categorical_dtype(dt)
                                      else 'string' if pat.is_string_dtype(dt)
                                      else 'datetime' if pat.is_datetime64_dtype(dt)
                                      else 'object')
        features = [Feature(i, col, to_feature_type(dtypes[i]))
                    for i, col in enumerate(self._ds.columns)]

        for f in features:
            col = self._ds.iloc[:, f.index]
            f.has_missing_values = col.hasnans
            # f.dtype = self._ds.dtypes[f.name]
            if f.is_categorical():
                f.values = self._unique_values(f.name)

        target = self._find_target_feature(features)
        self._set_feature_as_target(target)

        meta = dict(
            features=features,
            target=target
        )
        log.debug("Metadata for dataset %s: %s", self.path, meta)
        return meta

    @profile(logger=log)
    def load_data(self):
        self._ensure_loaded()
        return self._ds

    def release(self, properties=None):
        super().release(properties)
        self._ds = None

    def _unique_values(self, col_name: str):
        dt = self._ds.dtypes[col_name]
        return sorted(dt.categories.values if hasattr(dt, 'categories')
                      else self._ds[col_name].unique())


class FileConverter:
    format = None

    def __init__(self) -> None:
        super().__init__()

    def convert(self, split: FileDatasplit) -> str:
        sp = split_path(split._path)
        sp.extension = self.format
        target_path = path_from_split(sp)
        if not os.path.isfile(target_path):
            self._write_file(split.data, target_path)
        return target_path

    @abstractmethod
    def _write_file(self, df, path):
        pass


class ArffConverter(FileConverter):
    format = 'arff'

    def _write_file(self, df, path):
        name = split_path(path).basename
        description = f"Arff dataset file generated by automlbenchmark from {name}."
        attributes = [(c,
                       ('INTEGER' if pat.is_integer_dtype(dt)
                        else 'REAL' if pat.is_float_dtype(dt)
                        else 'NUMERIC' if pat.is_numeric_dtype(dt)
                        # numeric categories need to be str: https://github.com/renatopp/liac-arff/issues/126
                        else [str(cat) for cat in sorted(dt.categories.values)] if pat.is_categorical_dtype(dt)
                        else 'STRING'
                        ))
                      for c, dt in zip(df.columns, df.dtypes)]
        arff_data = dict(
            description=description,
            relation=name,
            attributes=attributes,
            data=df.values
        )
        with open(path, 'w') as file:
            arff.dump(arff_data, file)


class CsvConverter(FileConverter):
    format = 'csv'

    def _write_file(self, df, path):
        df.to_csv(path, header=True, index=False)


class ParquetConverter(FileConverter):
    format = 'parquet'

    def _write_file(self, df, path):
        df.to_parquet(path)


__file_converters__ = [ArffConverter, CsvConverter, ParquetConverter]


def _get_file_convert_cls(fmt=None):
    cls = next((fc for fc in __file_converters__ if fc.format == fmt), None)
    if cls is None:
        supported = [ds.format for ds in __file_converters__]
        raise ValueError(f"`{fmt}` is not among supported formats: {supported}.")
    return cls
