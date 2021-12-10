from abc import abstractmethod
import logging
import os
import re
import tempfile
from typing import List, Union

import arff
import numpy as np
import pandas as pd
import pandas.api.types as pat

from ..data import AuxData, Dataset, DatasetType, Datasplit, Feature, DF
from ..datautils import read_csv, to_data_frame
from ..resources import config as rconfig
from ..utils import Namespace as ns, as_list, lazy_property, list_all_files, memoize, path_from_split, profile, split_path

from .fileutils import is_archive, is_valid_url, unarchive_file, get_file_handler


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
        paths = self._extract_train_test_paths(dataset.path if 'path' in dataset else dataset, fold=fold)
        assert fold < len(paths['train']), f"No training dataset available for fold {fold} among dataset files {paths['train']}"
        # seed = rget().seed(fold)
        # if len(paths['test']) == 0:
        #     log.warning("No test file in the dataset, the train set will automatically be split 90%/10% using the given seed.")
        # else:
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
            return CsvDataset(train_path, test_path, target=target, features=features, type=type_)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    @profile(logger=log)
    def load_auxiliary_data(self, dataset, auxiliary_data, fold=0):
        auxiliary_data = auxiliary_data if isinstance(auxiliary_data, ns) else ns(path=auxiliary_data)
        log.debug("Loading auxiliary data %s", auxiliary_data)
        paths = self._extract_auxiliary_paths(auxiliary_data.path if 'path' in auxiliary_data else auxiliary_data, fold=fold)
        train_data = None
        test_data = None
        if 'train' in paths:
            train_path = paths['train'][fold]
            train_data = FileAuxData(train_path)
        if 'test' in paths:
            test_path = paths['test'][fold]
            test_data = FileAuxData(test_path)
        dataset._attach_auxiliary_data(train_data, test_data)
        return dataset

    def _extract_auxiliary_paths(self, auxiliary_data, fold=None):
        if isinstance(auxiliary_data, (tuple, list)):
            assert len(auxiliary_data) % 2 == 0, "auxiliary data list must contain an even number of paths: [train_auxiliary_0, test_auxiliary_0, train_auxiliary_1, test_auxiliary_1, ...]."
            return self._extract_paths(ns(train=[p for i, p in enumerate(auxiliary_data) if i % 2 == 0],
                                                   test=[p for i, p in enumerate(auxiliary_data) if i % 2 == 1]),
                                                fold=fold, train_suffix='train_auxiliary', test_suffix='test_auxiliary')
        elif isinstance(auxiliary_data, ns):
            return dict(
                train=[self._extract_paths(p, fold=fold, train_suffix='train_auxiliary', test_suffix='test_auxiliary')['train'][0]
                       if i == fold else None
                       for i, p in enumerate(as_list(auxiliary_data.train))] if 'train' in auxiliary_data else [],
                test=[self._extract_paths(p, fold=fold, train_suffix='train_auxiliary', test_suffix='test_auxiliary')['train'][0]
                      if i == fold else None
                      for i, p in enumerate(as_list(auxiliary_data.test))] if 'test' in auxiliary_data else []
            )
        else:
            self._extract_paths(auxiliary_data, fold=fold, train_suffix='train_auxiliary', test_suffix='test_auxiliary')


    def _extract_train_test_paths(self, dataset, fold=None):
        if isinstance(dataset, (tuple, list)):
            assert len(dataset) % 2 == 0, "dataset list must contain an even number of paths: [train_0, test_0, train_1, test_1, ...]."
            return self._extract_paths(ns(train=[p for i, p in enumerate(dataset) if i % 2 == 0],
                                                     test=[p for i, p in enumerate(dataset) if i % 2 == 1]),
                                                  fold=fold, train_suffix='train', test_suffix='test')
        elif isinstance(dataset, ns):
            return dict(train=[self._extract_paths(p, fold=fold, train_suffix='train', test_suffix='test')['train'][0]
                               if i == fold else None
                               for i, p in enumerate(as_list(dataset.train))],
                        test=[self._extract_paths(p, fold=fold, train_suffix='train', test_suffix='test')['train'][0]
                              if i == fold else None
                              for i, p in enumerate(as_list(dataset.test))])
        else:
            self._extract_paths(dataset, fold=fold, train_suffix='train', test_suffix='test')


    def _extract_paths(self, data, fold=None, train_suffix='train', test_suffix='test'):
        train_search_pat = re.compile(rf"(?:(.*)[_-]){train_suffix}(?:[_-](\d+))?\.\w+")
        test_search_pat = re.compile(rf"(?:(.*)[_-]){train_suffix}(?:[_-](\d+))?\.\w+")
        is_aux_data = False
        if train_suffix == 'train_auxiliary' and test_suffix == 'test_auxiliary':
            is_aux_data = True

        assert isinstance(data, str)
        data = os.path.expanduser(data)
        data = data.format(**rconfig().common_dirs)

        if os.path.exists(data):
            if os.path.isfile(data):
                # we leave the auxiliary data handling to the user
                if is_archive(data) and not is_aux_data:
                    arch_name, _ = os.path.splitext(os.path.basename(data))
                    dest_folder = os.path.join(self._cache_dir, arch_name)
                    if not os.path.exists(dest_folder):  # don't uncompress if previously done
                        dest_folder = unarchive_file(data, dest_folder)
                    return self._extract_paths(dest_folder, train_suffix=train_suffix, test_suffix=test_suffix)
                else:
                    return dict(train=[data], test=[])
            elif os.path.isdir(data):
                files = list_all_files(data)
                log.debug("Files found in data folder %s: %s", data, files)
                assert len(files) > 0, f"Empty folder: {data}"
                if len(files) == 1:
                    return dict(train=files, test=[])

                train_matches = [m for m in [train_search_pat.search(f) for f in files] if m]
                test_matches = [m for m in [test_search_pat.search(f) for f in files] if m]
                # verify they're for the same dataset (just based on name)
                if not is_aux_data:
                    assert train_matches and test_matches, f"Folder {data} must contain at least one training and one test dataset."
                else:
                    assert train_matches or test_matches, f"Folder {data} must contain at least one training auxiliary data or one test auxiliary data."
                root_names = {m[1] for m in (train_matches+test_matches)}
                assert len(root_names) == 1, f"All data files in {data} should follow the same naming: xxxxx_{train_suffix}_N.ext or xxxxx_{test_suffix}_N.ext with N starting from 0."

                train_no_fold = next((m[0] for m in train_matches if m[2] is None), None)
                test_no_fold = next((m[0] for m in test_matches if m[2] is None), None)
                if train_no_fold and test_no_fold:
                    return dict(train=[train_no_fold], test=[test_no_fold])

                paths = dict(train=[], test=[])
                fold = 0
                while fold >= 0:
                    train = next((m[0] for m in train_matches if m[2] == str(fold)), None)
                    test = next((m[0] for m in test_matches if m[2] == str(fold)), None)
                    if not is_aux_data:
                        if train and test:
                            paths['train'].append(train)
                            paths['test'].append(test)
                            fold += 1
                        else:
                            fold = -1
                    else:
                        if train:
                            paths['train'].append(train)
                        if test:
                            paths['test'].append(test)
                        if not train and not test:
                            fold = -1
                        fold += 1
                assert len(paths) > 0, f"No data file found in {data}: they should follow the naming xxxx_{train_suffix}.ext, xxxx_{test_suffix}.ext or xxxx_{train_suffix}_0.ext, xxxx_{test_suffix}_0.ext, xxxx_{train_suffix}_1.ext, ..."
                return paths
        elif is_valid_url(data):
            cached_file = os.path.join(self._cache_dir, os.path.basename(data))
            if not os.path.exists(cached_file):  # don't download if previously done
                handler = get_file_handler(data)
                assert handler.exists(data), f"Invalid path/url: {data}"
                handler.download(data, dest_path=cached_file)
            return self._extract_paths(cached_file, fold=fold, train_suffix=train_suffix, test_suffix=test_suffix)
        else:
            raise ValueError(f"Invalid dataset description: {data}")


class FileAuxData(AuxData):

    def __init__(self, path):
        super().__init__()
        self._path = path

    @property
    def path(self) -> str:
        return self._path

    @property
    def data(self) -> DF:
        return NotImplementedError


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

    def _attach_auxiliary_data(self, train_auxiliary_data, test_auxiliary_data):
        self._train._attach_auxiliary_data(train_auxiliary_data)
        self._test._attach_auxiliary_data(test_auxiliary_data)

    @memoize
    def _get_metadata(self, prop):
        meta = self._train.load_metadata()
        return meta[prop]


class FileDatasplit(Datasplit):

    def __init__(self, dataset: FileDataset, format: str, path: str):
        super().__init__(dataset, format)
        self._path = path
        self._data = {format: path}
        self._auxiliary_data = None


    def data_path(self, format):
        supported_formats = [cls.format for cls in __file_converters__]
        if format not in supported_formats:
            name = split_path(self._path).basename
            raise ValueError(f"Dataset {name} is only available in one of {supported_formats} formats.")
        return self._get_data(format)

    @property
    def has_auxiliary_data(self) -> bool:
        return self._auxiliary_data != None
       
    @property
    def auxiliary_data(self) -> AuxData:
        return self._auxiliary_data

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
                log.warning("Forcing target column %s as 'category' for classification problems: was originally detected as '%s'.",
                            target.name, target.data_type)
                # target.data_type = 'category'
        target.is_target = True

    def _attach_auxiliary_data(self, auxiliary_data):
        self._auxiliary_data = auxiliary_data


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
                 target=None, features=None, type=None):
        # todo: handle auto-split (if test_path is None): requires loading the training set, split, save
        super().__init__(CsvDatasplit(self, train_path), CsvDatasplit(self, test_path),
                         target=target, features=features, type=type)
        self._dtypes = None


class CsvDatasplit(FileDatasplit):

    def __init__(self, dataset, path):
        super().__init__(dataset, format='csv', path=path)
        self._ds = None

    def _ensure_loaded(self):
        if self._ds is None:
            if self.dataset._dtypes is None:
                df = read_csv(self.path)
                # df = df.convert_dtypes()
                dt_conversions = {name: 'category'
                                  for name, dtype in zip(df.dtypes.index, df.dtypes.values)
                                  if pat.is_string_dtype(dtype) or pat.is_object_dtype(dtype)}
                # we could be a bit more clever in the future and convert 'string' to category iff len(distinct values) << nrows
                if dt_conversions:
                    df = df.astype(dt_conversions, copy=False)

                self._ds = df
                self.dataset._dtypes = self._ds.dtypes
            else:
                self._ds = read_csv(self.path, dtype=self.dataset._dtypes.to_dict())

    @profile(logger=log)
    def load_metadata(self):
        self._ensure_loaded()
        dtypes = self.dataset._dtypes
        to_feature_type = lambda dt: ('int' if pat.is_integer_dtype(dt)
                                      else 'float' if pat.is_float_dtype(dt)
                                      else 'number' if pat.is_numeric_dtype(dt)
                                      else 'category' if pat.is_categorical_dtype(dt)
                                      else 'string' if pat.is_string_dtype(dt)
                                      # else 'datetime' if pat.is_datetime64_dtype(dt)
                                      else 'object')
        features = [Feature(i, col, to_feature_type(dtypes[i]))
                    for i, col in enumerate(self._ds.columns)]

        for f in features:
            col = self._ds.iloc[:, f.index]
            f.has_missing_values = col.hasnans
            if f.is_categorical():
                f.values = sorted(self._ds.dtypes[f.name].categories.values)

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
        with open(path, 'w') as file:
            description = f"Arff dataset file generated by automlbenchmark from {name}."
            attributes = [(c,
                           ('INTEGER' if pat.is_integer_dtype(dt)
                            else 'REAL' if pat.is_float_dtype(dt)
                            else 'NUMERIC' if pat.is_numeric_dtype(dt)
                            else sorted(dt.categories.values) if pat.is_categorical_dtype(dt)
                            else 'STRING'
                            ))
                          for c, dt in zip(df.columns, df.dtypes)]
            arff.dump(dict(
                description=description,
                relation=name,
                attributes=attributes,
                data=df.values
            ), file)


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
