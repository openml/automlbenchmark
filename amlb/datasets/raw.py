from abc import abstractmethod
import logging
import os
from typing import List

import arff
import numpy as np

from ..data import Dataset, DatasetType, Datasplit, Feature
from ..utils import lazy_property, obj_size, profile, to_mb


log = logging.getLogger(__name__)


class RawLoader:

    def load(self, path):
        pass


class RawDataset(Dataset):

    @property
    def type(self) -> DatasetType:
        pass

    @property
    def train(self) -> Datasplit:
        pass

    @property
    def test(self) -> Datasplit:
        pass

    @property
    def features(self) -> List[Feature]:
        pass

    @property
    def target(self) -> Feature:
        pass

    @abstractmethod
    def load_file(self, f):
        pass


class ArffDataset(RawDataset):

    def load_file(self, f):
        return arff.load(f)


class CsvDataset(RawDataset):

    def load_file(self, f):
        return np.loadtxt(f)

