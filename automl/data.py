from abc import ABC, abstractmethod
from typing import List

import numpy as np
from numpy import ndarray

from .utils import encoder, lazy_property, repr_def


class Feature:

    def __init__(self, index, name, data_type, values=None):
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

    def is_categorical(self, strict=True):
        if strict:
            return self.data_type is not None and self.data_type.lower() in ['categorical', 'nominal', 'enum']
        else:
            return self.data_type is not None and self.data_type.lower() not in ['numeric', 'integer', 'real']

    @lazy_property
    def label_encoder(self):
        return encoder(self.values, 'label')

    @lazy_property
    def label_binarizer(self):
        return encoder(self.values, 'binary')

    @lazy_property
    def one_hot_encoder(self):
        return encoder(self.values, 'one_hot')

    def encode(self, labelled_data, label_encoder=None):
        label_encoder = self.label_encoder if not label_encoder else label_encoder
        return label_encoder.transform(labelled_data) if label_encoder else labelled_data

    def decode(self, encoded_data, label_encoder=None):
        label_encoder = self.label_encoder if not label_encoder else label_encoder
        return label_encoder.inverse_transform(encoded_data) if label_encoder else encoded_data

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
        encoded_cols = [p.encode(self.data[:, p.index]) for p in self.dataset.predictors]
        return np.hstack(tuple(col.reshape(-1, 1) for col in encoded_cols))

    @lazy_property
    def y_enc(self) -> ndarray:
        return self.dataset.target.encode(self.y)


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

