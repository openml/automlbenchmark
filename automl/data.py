from abc import ABC, abstractmethod
from typing import List

import numpy as np
from numpy import ndarray

from .utils import Encoder, lazy_property, repr_def


class Feature:

    def __init__(self, index, name, data_type, values=None, is_target=False):
        """

        :param index:
        :param name:
        :param type:
        :param values:
        """
        self.index = index
        self.name = name
        self.data_type = data_type
        self.values = values
        self.is_target = is_target

    def is_categorical(self, strict=True):
        if strict:
            return self.data_type is not None and self.data_type.lower() in ['categorical', 'nominal', 'enum']
        else:
            return self.data_type is not None and self.data_type.lower() not in ['numeric', 'integer', 'real']

    @lazy_property
    def label_encoder(self):
        return Encoder('label' if self.values else 'no-op', target=self.is_target, missing_handle='mask').fit(self.values)

    @lazy_property
    def one_hot_encoder(self):
        return Encoder('one-hot' if self.values else 'no-op', target=self.is_target, missing_handle='mask').fit(self.values)

    def __repr__(self):
        return repr_def(self)


class Datasplit(ABC):

    def __init__(self, dataset, format):
        """

        :param format:
        """
        super().__init__()
        self.dataset = dataset
        self.format = format

    @property
    @abstractmethod
    def path(self) -> str:
        """

        :return:
        """
        pass

    @property
    @abstractmethod
    def data(self) -> ndarray:
        """

        :return:
        """
        pass

    @lazy_property
    def X(self) -> ndarray:
        """

        :return:
        """
        predictors_ind = [p.index for p in self.dataset.predictors]
        return self.data[:, predictors_ind]

    @lazy_property
    def y(self) -> ndarray:
        """

        :return:
        """
        return self.data[:, self.dataset.target.index]

    @lazy_property
    def X_enc(self) -> ndarray:
        # todo: should we use one_hot_encoder here instead?
        encoded_cols = [p.label_encoder.transform(self.data[:, p.index]) for p in self.dataset.predictors]
        return np.hstack(tuple(col.reshape(-1, 1) for col in encoded_cols))

    @lazy_property
    def y_enc(self) -> ndarray:
        return self.dataset.target.label_encoder.transform(self.y)


class Dataset(ABC):

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def train(self) -> Datasplit:
        """

        :return:
        """
        pass

    @property
    @abstractmethod
    def test(self) -> Datasplit:
        """

        :return:
        """
        pass

    @property
    @abstractmethod
    def features(self) -> List[Feature]:
        """

        :return:
        """
        pass

    @property
    def predictors(self) -> List[Feature]:
        """

        :return:
        """
        return [f for f in self.features if f.name != self.target.name]

    @property
    @abstractmethod
    def target(self) -> Feature:
        """

        :return:
        """
        pass

