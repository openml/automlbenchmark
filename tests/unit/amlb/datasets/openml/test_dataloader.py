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


def test_load_binary_task(oml_loader):
    fold = random.randint(0, 9)
    ds = oml_loader.load(task_id=3913, fold=fold)  # kc2
    assert ds.type is DatasetType.binary
    assert len(ds.features) == 22
    assert len(ds.predictors) == 21

    assert ds.target.name == "problems"
    assert len(ds.target.values) == 2
    assert ds.target.values == ["no", "yes"]
    assert ds.target.data_type == 'category'
    assert ds.target.is_target
    assert not ds.target.has_missing_values

    assert all([p.data_type == 'number' for p in ds.predictors])
    assert all([p.values is None for p in ds.predictors])
    assert not any([p.is_target for p in ds.predictors])
    assert not any([p.has_missing_values for p in ds.predictors])

    ds_id = ds._oml_dataset.dataset_id
    assert ds.train.path.endswith(os.path.join("datasets", str(ds_id), f"dataset_train_{fold}.arff"))
    assert ds.test.path.endswith(os.path.join("datasets", str(ds_id), f"dataset_test_{fold}.arff"))
    assert ds.train.data_path('csv').endswith(os.path.join("datasets", str(ds_id), f"dataset_train_{fold}.csv"))
    assert ds.test.data_path('csv').endswith(os.path.join("datasets", str(ds_id), f"dataset_test_{fold}.csv"))

    assert isinstance(ds.train.X, pd.DataFrame)
    assert isinstance(ds.train.y, pd.DataFrame)
    assert isinstance(ds.train.X_enc, np.ndarray)
    assert isinstance(ds.train.y_enc, np.ndarray)

    assert len(ds.train.X) == len(ds.train.y)
    assert len(ds.train.X.columns) == len(ds.predictors)
    assert len(ds.train.y.columns) == 1
    assert ds.train.y.columns == [ds.target.name]
    assert len(ds.train.X) > len(ds.test.X)
    assert ds.train.X.dtypes.apply(lambda dt: pd.api.types.is_float_dtype(dt)).all()
    assert pd.api.types.is_categorical_dtype(ds.train.y.dtypes.iloc[0])
    assert ds.test.X.dtypes.equals(ds.train.X.dtypes)
    assert ds.test.y.dtypes.equals(ds.train.y.dtypes)

    assert ds.train.X_enc.shape == ds.train.X.shape
    assert np.issubdtype(ds.train.X_enc.dtype, np.floating)
    assert np.issubdtype(ds.train.y_enc.dtype, np.floating)  # not ideal given that it's binary, but well…


def test_load_multiclass_task(oml_loader):
    fold = random.randint(0, 9)
    ds = oml_loader.load(task_id=59, fold=fold)  # iris
    assert ds.type is DatasetType.multiclass

    assert len(ds.features) == 5
    assert len(ds.predictors) == 4

    assert ds.target.name == "class"
    assert len(ds.target.values) == 3
    assert ds.target.values == ["iris-setosa", "iris-versicolor", "iris-virginica"]
    assert ds.target.data_type == 'category'
    assert ds.target.is_target
    assert not ds.target.has_missing_values

    assert all([p.data_type == 'number' for p in ds.predictors])
    assert all([p.values is None for p in ds.predictors])
    assert not any([p.is_target for p in ds.predictors])
    assert not any([p.has_missing_values for p in ds.predictors])

    ds_id = ds._oml_dataset.dataset_id
    assert ds.train.path.endswith(os.path.join("datasets", str(ds_id), f"dataset_train_{fold}.arff"))
    assert ds.test.path.endswith(os.path.join("datasets", str(ds_id), f"dataset_test_{fold}.arff"))
    assert ds.train.data_path('csv').endswith(os.path.join("datasets", str(ds_id), f"dataset_train_{fold}.csv"))
    assert ds.test.data_path('csv').endswith(os.path.join("datasets", str(ds_id), f"dataset_test_{fold}.csv"))

    assert isinstance(ds.train.X, pd.DataFrame)
    assert isinstance(ds.train.y, pd.DataFrame)
    assert isinstance(ds.train.X_enc, np.ndarray)
    assert isinstance(ds.train.y_enc, np.ndarray)

    assert len(ds.train.X) > len(ds.test.X)
    assert len(ds.train.X) == len(ds.train.y)
    assert len(ds.train.X.columns) == len(ds.predictors)
    assert len(ds.train.y.columns) == 1
    assert ds.train.y.columns == [ds.target.name]
    assert len(ds.train.X) > len(ds.test.X)
    assert ds.train.X.dtypes.apply(lambda dt: pd.api.types.is_float_dtype(dt)).all()
    assert pd.api.types.is_categorical_dtype(ds.train.y.dtypes.iloc[0])
    assert ds.test.X.dtypes.equals(ds.train.X.dtypes)
    assert ds.test.y.dtypes.equals(ds.train.y.dtypes)

    assert ds.train.X_enc.shape == ds.train.X.shape
    assert np.issubdtype(ds.train.X_enc.dtype, np.floating)
    assert np.issubdtype(ds.train.y_enc.dtype, np.floating)


def test_load_regression_task(oml_loader):
    fold = random.randint(0, 9)
    ds = oml_loader.load(task_id=2295, fold=fold)  # cholesterol
    assert ds.type is DatasetType.regression

    assert len(ds.features) == 14
    assert len(ds.predictors) == 13

    assert ds.target.name == "chol"
    assert ds.target.values is None
    assert ds.target.data_type == 'number'
    assert ds.target.is_target
    assert not ds.target.has_missing_values

    assert len([p for p in ds.predictors if p.data_type == 'number']) == 6
    assert len([p for p in ds.predictors if p.data_type == 'category']) == 7
    assert not any([p.is_target for p in ds.predictors])
    assert len([p for p in ds.predictors if p.has_missing_values]) == 2

    ds_id = ds._oml_dataset.dataset_id
    assert ds.train.path.endswith(os.path.join("datasets", str(ds_id), f"dataset_train_{fold}.arff"))
    assert ds.test.path.endswith(os.path.join("datasets", str(ds_id), f"dataset_test_{fold}.arff"))
    assert ds.train.data_path('csv').endswith(os.path.join("datasets", str(ds_id), f"dataset_train_{fold}.csv"))
    assert ds.test.data_path('csv').endswith(os.path.join("datasets", str(ds_id), f"dataset_test_{fold}.csv"))

    assert isinstance(ds.train.X, pd.DataFrame)
    assert isinstance(ds.train.y, pd.DataFrame)
    assert isinstance(ds.train.X_enc, np.ndarray)
    assert isinstance(ds.train.y_enc, np.ndarray)

    assert len(ds.train.X) > len(ds.test.X)
    assert len(ds.train.X) == len(ds.train.y)
    assert len(ds.train.X.columns) == len(ds.predictors)
    assert len(ds.train.y.columns) == 1
    assert ds.train.y.columns == [ds.target.name]
    assert len(ds.train.X) > len(ds.test.X)
    categoricals = [p.name for p in ds.predictors if p.data_type == 'category']
    numericals = [p.name for p in ds.predictors if p.name not in categoricals]
    assert ds.train.X.dtypes.filter(items=categoricals).apply(lambda dt: pd.api.types.is_categorical_dtype(dt)).all()
    assert ds.train.X.dtypes.filter(items=numericals).apply(lambda dt: pd.api.types.is_float_dtype(dt)).all()
    assert pd.api.types.is_float_dtype(ds.train.y.dtypes.iloc[0])
    assert ds.test.X.dtypes.equals(ds.train.X.dtypes)
    assert ds.test.y.dtypes.equals(ds.train.y.dtypes)

    assert ds.train.X_enc.shape == ds.train.X.shape
    assert np.issubdtype(ds.train.X_enc.dtype, np.floating)   # all categorical features are directly encoded as float
    assert np.issubdtype(ds.train.y_enc.dtype, np.floating)

