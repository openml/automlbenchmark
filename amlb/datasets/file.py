from abc import abstractmethod
import logging
import os
import re
import shutil
import tarfile
import tempfile
from typing import List, Union
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import zipfile

import arff
import numpy as np

from ..data import Dataset, DatasetType, Datasplit, Feature
from ..datautils import read_csv, to_data_frame
from ..utils import Namespace, as_list, lazy_property, list_all_files, profile, touch


log = logging.getLogger(__name__)

VALID_URLS = ("http", "https")


def is_valid_url(url):
    return urlparse(url).scheme in VALID_URLS


def url_exists(url):
    if not is_valid_url(url):
        return False
    head_req = Request(url, method='HEAD')
    try:
        with urlopen(head_req) as test:
            return test.status == 200
    except URLError:
        return False


def download_file(url, dest_path):
    touch(dest_path, as_dir=True)
    with urlopen(url) as resp, open(dest_path, 'w') as dest:
        shutil.copyfileobj(resp, dest)


def is_archive(path):
    return zipfile.is_zipfile(path) or tarfile.is_tarfile(path)


def unarchive_file(path, dest_folder=None):
    # dest = dest_folder if dest_folder else os.path.dirname(path)
    dest = dest_folder if dest_folder else os.path.splitext(path)
    touch(dest, as_dir=True)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            zf.extractall(path=dest_folder)
    elif tarfile.is_tarfile(path):
        with tarfile.TarFile(path) as tf:
            tf.extractall(path=dest_folder)
    return dest


class FileLoader:

    def __init__(self, cache_dir=None):
        self._cache_dir = cache_dir if cache_dir else tempfile.mkdtemp(prefix='amlb_cache')

    def load(self, dataset, fold=0):
        log.debug("Loading dataset %s", dataset)
        paths = self._extract_train_test_paths(dataset)
        target = dataset.target if isinstance(dataset, (dict, Namespace)) and 'target' in dataset else None
        type_ = dataset.type if  isinstance(dataset, (dict, Namespace)) and 'type' in dataset else None
        ext = os.path.splitext(paths['train'][0])[1].lower()
        train_path = paths['train'][fold]
        test_path = paths['test'][fold] if 'test' in paths and len(paths['test']) > 0 else None
        log.info(f"Using training set {train_path} with test set {test_path}.")
        if ext == '.arff':
            return ArffDataset(train_path, test_path, target=target, type=type_)
        elif ext == '.csv':
            return CsvDataset(train_path, test_path, target=target, type=type_)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _extract_train_test_paths(self, dataset):
        if isinstance(dataset, (tuple, list)):
            assert len(dataset) % 2 == 0, "dataset list must contain an even number of paths: [train_0, test_0, train_1, test_1, ...]."
            return self._extract_train_test_paths(dict(train=[p for i, p in enumerate(dataset) if p % 2 == 0],
                                                       test=[p for i, p in enumerate(dataset) if p % 2 == 1]))
        elif isinstance(dataset, (dict, Namespace)):
            return dict(train=[self._extract_train_test_paths(p)['train'][0] for p in as_list(dataset['train'])],
                        test=[self._extract_train_test_paths(p)['train'][0] for p in as_list(dataset['test'] if 'test' in dataset else [])])
        else:
            assert isinstance(dataset, str)
            dataset = os.path.expanduser(dataset)

        if os.path.exists(dataset):
            if os.path.isfile(dataset):
                if is_archive(dataset):
                    arch_name, _ = os.path.splitext(os.path.basename(dataset))
                    dest_folder = unarchive_file(dataset, os.path.join(self._cache_dir, arch_name))
                    return self._extract_train_test_paths(dest_folder)
                else:
                    return dict(train=[dataset])
            elif os.path.isdir(dataset):
                files = list_all_files(dataset)
                log.debug("Files found in dataset folder %s: %s", dataset, files)
                assert len(files) > 0, f"Empty folder: {dataset}"
                if len(files) == 1:
                    return dict(train=files)

                train_matches = [m for m in [re.search(r"(?:(.*)_)train(?:_(\d+))?\.\w+", f) for f in files] if m]
                test_matches = [m for m in [re.search(r"(?:(.*)_)test(?:_(\d+))?\.\w+", f) for f in files] if m]
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
            if os.path.exists(cached_file):
                return self._extract_train_test_paths(cached_file)
            assert url_exists(dataset), f"Invalid path/url: {dataset}"
            download_file(dataset, cached_file)
            return self._extract_train_test_paths(cached_file)
        else:
            raise ValueError(f"Invalid dataset description: {dataset}")


class FileDataset(Dataset):

    def __init__(self, train: Datasplit, test: Datasplit, target: str = None, type: str = None):
        super().__init__()
        self._train = train
        self._test = test
        self._target = target
        self._type = type
        self._metadata = None

    @property
    def type(self) -> DatasetType:
        return DatasetType[self._type] if self._type is not None else self._get_metadata('type')

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
        if not self._metadata:
            self._metadata = self._train.load_metadata()
        return self._metadata[prop]


class FileDatasplit(Datasplit):

    def __init__(self, dataset: FileDataset, format: str, path: str):
        super().__init__(dataset, format)
        self._path = path

    @property
    def path(self):
        return self._path

    @lazy_property
    @profile(logger=log)
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


class ArffDataset(FileDataset):

    def __init__(self, train_path, test_path, target=None, type=None):
        # todo: handle auto-split (if test_path is None): requires loading the training set, split, save
        super().__init__(ArffDatasplit(self, train_path), ArffDatasplit(self, test_path), target=target, type=type)


class ArffDatasplit(FileDatasplit):

    def __init__(self, dataset, path):
        super().__init__(dataset, format='arff', path=path)

    def load_metadata(self):
        ds = arff.load(self.path)
        attrs = ds['attributes']
        # arff loader types = ['NUMERIC', 'REAL', 'INTEGER', 'STRING']
        to_feature_type = lambda arff_type: 'categorical' if arff_type.lower() == 'string' else arff_type.lower()

        features = [
            Feature(
                i,
                attr[0],
                to_feature_type(attr[1]),
            )
            for i, attr in enumerate(attrs)
        ]
        target = self._find_target_feature(features)
        target.is_target = True

        df = to_data_frame(ds['data'])
        for f in features:
            col = df.iloc[:, f.index]
            f.has_missing_values = col.hasnans
            if f.is_categorical():
                unique_values = col.dropna().unique() if f.has_missing_values else col.unique()
                f.values = sorted(unique_values)

        dataset_type = DatasetType.regression
        if target.is_categorical():
            dataset_type = DatasetType.binary if len(target.values) == 2 else DatasetType.multiclass

        return dict(
            type=dataset_type,
            features=features,
            target=target
        )

    def load_data(self):
        ds = arff.load(self.path)
        return np.asarray(ds['data'], dtype=object)


class CsvDataset(FileDataset):

    def __init__(self, train_path, test_path, target=None, type=None):
        # todo: handle auto-split (if test_path is None): requires loading the training set, split, save
        super().__init__(CsvDatasplit(self, train_path), CsvDatasplit(self, test_path), target=target, type=type)


class CsvDatasplit(FileDatasplit):

    def __init__(self, dataset, path):
        super().__init__(dataset, format='csv', path=path)

    def load_metadata(self):
        # df = np.genfromtxt(self.path, dtype=None, names=True)
        df = read_csv(self.path, dtype=object)
        dtypes = df.dtypes
        to_feature_type = lambda dtype: ('categorical' if np.issubdtype(dtype, np.object_)
                                         else 'integer' if np.issubdtype(dtype, np.integer)
                                         else 'real' if np.issubdtype(dtype, np.floating)
                                         else 'numeric')

        features = [
            Feature(
                i,
                col,
                to_feature_type(dtypes[i])
            )
            for i, col in enumerate(df.columns)
        ]
        target = self._find_target_feature(features)
        target.is_target = True

        for f in features:
            col = df.iloc[:, f.index]
            f.has_missing_values = col.hasnans
            if f.is_categorical():
                unique_values = col.dropna().unique() if f.has_missing_values else col.unique()
                f.values = sorted(unique_values)

        dataset_type = DatasetType.regression
        if target.is_categorical():
            dataset_type = DatasetType.binary if len(target.values) == 2 else DatasetType.multiclass

        return dict(
            type=dataset_type,
            features=features,
            target=target
        )

    def load_data(self):
        # return np.genfromtxt(f, dtype=None)
        return read_csv(self.path, as_data_frame=False, dtype=object)

