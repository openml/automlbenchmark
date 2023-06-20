import os
import random

import numpy as np
import pandas as pd
import pytest

from amlb.resources import from_config
from amlb.data import DatasetType
from amlb.datasets.openml import OpenmlLoader
from amlb.utils import Namespace as ns


@pytest.fixture
def oml_config():
    return from_config(
        ns(
            input_dir="my_input",
            output_dir="my_output",
            user_dir="my_user_dir",
            root_dir="my_root_dir",

            openml=ns(
                apikey="c1994bdb7ecb3c6f3c8f3b35f4b47f1f",
                infer_dtypes=False
            )
        )
    ).config


@pytest.fixture
def oml_loader(oml_config):
    return OpenmlLoader(api_key=oml_config.openml.apikey)


@pytest.mark.use_disk
@pytest.mark.use_web
def test_load_binary_task(oml_loader):
    fold = random.randint(0, 9)
    ds = oml_loader.load(task_id=3913, fold=fold)  # kc2
    assert ds.type is DatasetType.binary
    _assert_X_y_types(ds.train)
    _assert_data_consistency(ds)
    _assert_data_paths(ds, ds._oml_dataset.dataset_id, fold)
    _assert_kc2_features(ds)


def _assert_kc2_features(dataset):
    assert len(dataset.features) == 22
    assert len(dataset.predictors) == 21

    _assert_target(dataset.target, "problems", ["no", "yes"])

    assert all([p.data_type == 'number' for p in dataset.predictors])
    assert all([p.values is None for p in dataset.predictors])
    assert not any([p.has_missing_values for p in dataset.predictors])

    assert dataset.train.X.dtypes.apply(lambda dt: pd.api.types.is_numeric_dtype(dt)).all()
    assert len(dataset.train.X.select_dtypes(include=['float']).columns) == 18
    assert len(dataset.train.X.select_dtypes(include=['uint8']).columns) == 3
    assert pd.api.types.is_categorical_dtype(dataset.train.y.dtypes.iloc[0])


@pytest.mark.use_disk
@pytest.mark.use_web
def test_load_multiclass_task(oml_loader):
    fold = random.randint(0, 9)
    ds = oml_loader.load(task_id=59, fold=fold)  # iris
    assert ds.type is DatasetType.multiclass
    _assert_X_y_types(ds.train)
    _assert_data_consistency(ds)
    _assert_data_paths(ds, ds._oml_dataset.dataset_id, fold)
    _assert_iris_features(ds)


def _assert_iris_features(dataset):
    assert len(dataset.features) == 5
    assert len(dataset.predictors) == 4

    _assert_target(dataset.target, "class", ["iris-setosa", "iris-versicolor", "iris-virginica"])

    assert all([p.data_type == 'number' for p in dataset.predictors])
    assert all([p.values is None for p in dataset.predictors])
    assert not any([p.has_missing_values for p in dataset.predictors])

    assert dataset.train.X.dtypes.apply(lambda dt: pd.api.types.is_float_dtype(dt)).all()
    assert pd.api.types.is_categorical_dtype(dataset.train.y.dtypes.iloc[0])


@pytest.mark.use_disk
@pytest.mark.use_web
def test_load_regression_task(oml_loader):
    fold = random.randint(0, 9)
    ds = oml_loader.load(task_id=2295, fold=fold)  # cholesterol
    assert ds.type is DatasetType.regression
    _assert_X_y_types(ds.train)
    _assert_data_consistency(ds)
    _assert_data_paths(ds, ds._oml_dataset.dataset_id, fold)
    _assert_cholesterol_features(ds)


def _assert_cholesterol_features(dataset):
    assert len(dataset.features) == 14
    assert len(dataset.predictors) == 13

    _assert_target(dataset.target, "chol")

    numericals = [p.name for p in dataset.predictors if p.data_type == 'number']
    categoricals = [p.name for p in dataset.predictors if p.data_type == 'category']
    assert len(numericals) == 6
    assert len(categoricals) == 7
    assert len([p for p in dataset.predictors if p.has_missing_values]) == 2

    assert dataset.train.X.dtypes.filter(items=numericals).apply(lambda dt: pd.api.types.is_numeric_dtype(dt)).all()
    assert dataset.train.X.dtypes.filter(items=categoricals).apply(lambda dt: pd.api.types.is_categorical_dtype(dt)).all()
    assert len(dataset.train.X.select_dtypes(include=['float']).columns) == 6
    assert pd.api.types.is_float_dtype(dataset.train.y.dtypes.iloc[0])


def _assert_target(target, name, values=None):
    assert target.name == name
    assert target.values == values
    assert target.data_type == 'category' if values else 'number'
    assert target.is_target
    assert not target.has_missing_values


def _assert_data_paths(dataset, ds_id, fold):
    assert dataset.train.path.endswith(os.path.join("datasets", str(ds_id), f"dataset_train_{fold}.arff"))
    assert dataset.test.path.endswith(os.path.join("datasets", str(ds_id), f"dataset_test_{fold}.arff"))
    assert dataset.train.data_path('csv').endswith(os.path.join("datasets", str(ds_id), f"dataset_train_{fold}.csv"))
    assert dataset.test.data_path('csv').endswith(os.path.join("datasets", str(ds_id), f"dataset_test_{fold}.csv"))
    assert dataset.train.data_path('parquet').endswith(os.path.join("datasets", str(ds_id), f"dataset_train_{fold}.parquet"))
    assert dataset.test.data_path('parquet').endswith(os.path.join("datasets", str(ds_id), f"dataset_test_{fold}.parquet"))


def _assert_X_y_types(data_split):
    assert isinstance(data_split.X, pd.DataFrame)
    assert isinstance(data_split.y, pd.DataFrame)
    assert isinstance(data_split.X_enc, np.ndarray)
    assert isinstance(data_split.y_enc, np.ndarray)


def _assert_data_consistency(dataset):
    assert len(dataset.train.X) == len(dataset.train.y)
    assert len(dataset.train.X.columns) == len(dataset.predictors)
    assert len(dataset.train.y.columns) == 1
    assert dataset.train.y.columns == [dataset.target.name]
    assert len(dataset.train.X) > len(dataset.test.X)

    assert not any([p.is_target for p in dataset.predictors])

    assert dataset.train.X_enc.shape == dataset.train.X.shape

    assert dataset.test.X.dtypes.equals(dataset.train.X.dtypes)
    assert dataset.test.y.dtypes.equals(dataset.train.y.dtypes)

    assert np.issubdtype(dataset.train.X_enc.dtype, np.floating)   # all categorical features are directly encoded as float
    assert np.issubdtype(dataset.train.y_enc.dtype, np.floating)

