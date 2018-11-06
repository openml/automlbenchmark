import os
import re

import openml as oml
import arff
import numpy as np

from .data import Dataset, Datasplit, Feature
from .utils import lazy_property


class Openml():

    def __init__(self, api_key):
        super().__init__()
        oml.config.apikey = api_key #todo, rely on default openml setup, apikey is private...

    def load(self, task_id, fold=0):
        task = oml.tasks.get_task(task_id)
        dataset = task.get_dataset()
        return OpenmlDataset(task, dataset, fold)


class OpenmlDataset(Dataset):

    def __init__(self, oml_task: oml.OpenMLTask, oml_dataset: oml.OpenMLDataset, fold=0):
        super().__init__()
        self._oml_task = oml_task
        self._oml_dataset = oml_dataset
        self.fold = fold
        self._train = None
        self._test = None
        self._attributes = None
        self._unique_values = {}

    @property
    def train(self):
        self._ensure_loaded()
        return self._train

    @property
    def test(self):
        self._ensure_loaded()
        return self._test

    @lazy_property
    def features(self):
        def get_values(f):
            """
            workaround to retrieve nominal values from arff file as openml (version 0.7.0) doesn't support yet
            retrieval of nominal values from the features.xml file
            :param f: openml feature
            :return: an array with nominal values
            """
            if f.data_type == 'nominal' and not f.nominal_values:
                f.nominal_values = next(values for name, values in self.attributes if name.lower() == f.name.lower())
            if not f.nominal_values:
                f.nominal_values = self._unique_values.get(f.name)
            return f.nominal_values

        return [Feature(f.index, f.name, f.data_type, get_values(f)) for i, f in sorted(self._oml_dataset.features.items())]

    @lazy_property
    def target(self):
        return next(f for f in self.features if f.name == self._oml_task.target_name)

    @property
    def attributes(self):
        if not self._attributes:
            with open(self._oml_dataset.data_file) as f:
                ds = arff.load(f)
                self._attributes = ds['attributes']
        return self._attributes

    def _ensure_loaded(self):
        if self._train is None and self._test is None:
            self._load_split()

    def _load_split(self):
        ds_path = self._oml_dataset.data_file
        train_path = _get_split_path_for_dataset(ds_path, 'train', self.fold)
        test_path = _get_split_path_for_dataset(ds_path, 'test', self.fold)

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            self._prepare_split_data(train_path, test_path)

        self._train = OpenmlDatasplit(self, train_path)
        self._test = OpenmlDatasplit(self, test_path)

    def _prepare_split_data(self, train_path, test_path):
        train_ind, test_ind = self._oml_task.get_train_test_split_indices(self.fold)
        #X, y = self._oml_task.get_X_and_y() #numpy arrays
        ods = self._oml_dataset

        # X, y, attr_is_categorical, attr_names = ods.get_data(self.target,
        #                                                    return_categorical_indicator=True,
        #                                                    return_attribute_names=True)
        # ods.retrieve_class_labels(self.target)

        with open(ods.data_file) as f:
            ds = arff.load(f)
        self._attributes = ds['attributes']
        self._extract_unique_values(ds)

        name_pattern = "{name}_{{split}}_{fold}".format(name=ods.name, fold=self.fold)
        _save_split_set(path=train_path,
                        name=name_pattern.format(split='train'),
                        full_dataset=ds,
                        indexes=train_ind)
        _save_split_set(path=test_path,
                        name=name_pattern.format(split='test'),
                        full_dataset=ds,
                        indexes=test_ind)

    def _extract_unique_values(self, arff_dataset):
        #TODO: support encoded string columns?
        pass



class OpenmlDatasplit(Datasplit):

    def __init__(self, dataset: Dataset, path:str):
        super().__init__(dataset, 'arff')
        self._path = path

    @property
    def path(self):
        return self._path

    @lazy_property
    def data(self):
        # use codecs for unicode support: path = codecs.load(self._path, 'rb', 'utf-8')
        with open(self.path) as file:
            ds = arff.load(file)
        return np.asarray(ds['data'])


def _get_split_path_for_dataset(ds_path, split='train', fold=0):
    ds_dir, ds_base = os.path.split(ds_path)
    split_base = re.sub(r'\.(\w+)$', r'_{split}_{fold}.\1'.format(split=split, fold=fold), ds_base)
    split_path = os.path.join(ds_dir, split_base)
    return split_path

def _save_split_set(path, name, full_dataset=None, indexes=None):
    # X_split = X[indexes, :]
    # y_split = y.reshape(-1, 1)[indexes, :]
    with open(path, 'w') as file:
        split_data = np.asarray(full_dataset['data'])[indexes, :]
        arff.dump({
            'description': full_dataset['description'],
            'relation': name,
            'attributes': full_dataset['attributes'],
            'data': split_data
        }, file)


