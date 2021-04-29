from abc import abstractmethod
import logging
import os
import re
import tempfile
from typing import List, Union

import arff
import numpy as np

from ..data import Dataset, DatasetType, Datasplit, Feature
from ..datautils import read_csv, to_data_frame
from ..resources import config as rconfig
from ..utils import Namespace as ns, as_list, cached, lazy_property, list_all_files, profile, split_path

from .fileutils import *


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

    def _extract_train_test_paths(self, dataset, fold=None):
        if isinstance(dataset, (tuple, list)):
            assert len(dataset) % 2 == 0, "dataset list must contain an even number of paths: [train_0, test_0, train_1, test_1, ...]."
            return self._extract_train_test_paths(ns(train=[p for i, p in enumerate(dataset) if i % 2 == 0],
                                                     test=[p for i, p in enumerate(dataset) if i % 2 == 1]),
                                                  fold=fold)
        elif isinstance(dataset, ns):
            return dict(train=[self._extract_train_test_paths(p)['train'][0]
                               if i == fold else None
                               for i, p in enumerate(as_list(dataset.train))],
                        test=[self._extract_train_test_paths(p)['train'][0]
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
            cached_file = os.path.join(self._cache_dir, os.path.basename(dataset))
            if not os.path.exists(cached_file):  # don't download if previously done
                assert url_exists(dataset), f"Invalid path/url: {dataset}"
                download_file(dataset, cached_file)
            return self._extract_train_test_paths(cached_file)
        else:
            raise ValueError(f"Invalid dataset description: {dataset}")


class FileDataset(Dataset):

    def __init__(self, train: Datasplit, test: Datasplit,
                 target: str = None, features: List[Union[ns, str]] = None, type: str = None):
        super().__init__()
        self._train = train
        self._test = test
        self._target = target
        self._features = features
        self._type = type

    @property
    def type(self) -> DatasetType:
        return (DatasetType[self._type] if self._type is not None
                else (DatasetType.binary if len(self.target.values) == 2 else DatasetType.multiclass) if self.target.is_categorical()
                else DatasetType.regression)

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

    def _get_metadata(self, prop):
        meta = self._train.load_metadata()
        return meta[prop]


class FileDatasplit(Datasplit):

    def __init__(self, dataset: FileDataset, format: str, path: str):
        super().__init__(dataset, format)
        self._path = path

    def data_path(self, format):
        if format != self.format:
            name = split_path(self._path).basename
            raise ValueError(f"Dataset {name} is only available in {self.format} format.")
        return self._path

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
                log.warning("Forcing target column %s as 'categorical' for classification problems: was originally detected as '%s'.", target.name, target.data_type)
                target.data_type = 'categorical'
        target.is_target = True


class ArffDataset(FileDataset):

    def __init__(self, train_path, test_path,
                 target=None, features=None, type=None):
        # todo: handle auto-split (if test_path is None): requires loading the training set, split, save
        super().__init__(ArffDatasplit(self, train_path), ArffDatasplit(self, test_path),
                         target=target, features=features, type=type)


class ArffDatasplit(FileDatasplit):

    def __init__(self, dataset, path):
        super().__init__(dataset, format='arff', path=path)

    @cached
    @profile(logger=log)
    def load_metadata(self):
        with open(self.path) as f:
            ds = arff.load(f)
        attrs = ds['attributes']
        # arff loader types = ['NUMERIC', 'REAL', 'INTEGER', 'STRING']
        to_feature_type = lambda arff_type: ('category' if isinstance(arff_type, (list, set))
                                             else 'string' if arff_type.lower() == 'string'
                                             else 'int' if arff_type.lower() == 'integer'
                                             else 'float' if arff_type.lower() == 'real'
                                             else 'number' if arff_type.lower() == 'numeric'
                                             else 'object')
        features = [
            Feature(
                i,
                attr[0],
                to_feature_type(attr[1]),
            )
            for i, attr in enumerate(attrs)
        ]
        target = self._find_target_feature(features)
        self._set_feature_as_target(target)

        df = to_data_frame(ds['data'])
        for f in features:
            col = df.iloc[:, f.index]
            f.has_missing_values = col.hasnans
            if f.is_categorical():
                arff_type = attrs[f.index][1]
                unique_values = (arff_type if isinstance(arff_type, (list, set))
                                 else col.dropna().unique() if f.has_missing_values
                                 else col.unique())
                f.values = sorted(unique_values)

        meta = dict(
            features=features,
            target=target
        )
        log.debug("Metadata for dataset %s: %s", self.path, meta)
        return meta

    @cached
    @profile(logger=log)
    def load_data(self):
        with open(self.path) as f:
            ds = arff.load(f)
        return np.asarray(ds['data'], dtype=object)


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

    def _set_feature_as_target(self, target: Feature):
        super()._set_feature_as_target(target)
        if target.is_categorical():
            self.dataset._dtypes[target.index] = np.object_

    @cached
    @profile(logger=log)
    def load_metadata(self):
        df = read_csv(self.path).convert_dtypes()
        self.dataset._dtypes = dtypes = df.dtypes
        to_feature_type = lambda dtype: ('object' if np.issubdtype(dtype, np.object_)
                                         else 'int' if np.issubdtype(dtype, np.integer)
                                         else 'float' if np.issubdtype(dtype, np.floating)
                                         else 'number')
        features = [
            Feature(
                i,
                col,
                to_feature_type(dtypes[i])
            )
            for i, col in enumerate(df.columns)
        ]
        target = self._find_target_feature(features)
        self._set_feature_as_target(target)

        for f in features:
            col = df.iloc[:, f.index]
            f.has_missing_values = col.hasnans
            if f.is_categorical():
                unique_values = col.dropna().unique() if f.has_missing_values else col.unique()
                f.values = [str(v) for v in sorted(unique_values)]

        meta = dict(
            features=features,
            target=target
        )
        log.debug("Metadata for dataset %s: %s", self.path, meta)
        return meta

    @cached
    @profile(logger=log)
    def load_data(self):
        # return np.genfromtxt(f, dtype=None)
        dtypes = self.dataset._dtypes.to_dict()
        data = read_csv(self.path, as_data_frame=False, dtype=dtypes)
        return data

