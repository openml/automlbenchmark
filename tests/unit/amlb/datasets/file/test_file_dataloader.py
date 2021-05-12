import os

import numpy as np
import pandas as pd
import pytest

from amlb.resources import from_config
from amlb.data import DatasetType
from amlb.datasets.file import FileLoader
from amlb.utils import Namespace as ns

here = os.path.realpath(os.path.dirname(__file__))
res = os.path.join(here, 'resources')


@pytest.fixture(autouse=True)
def file_config():
    return from_config(
        ns(
            input_dir="my_input",
            output_dir="my_output",
            user_dir="my_user_dir",
            root_dir="my_root_dir",
        )
    ).config


@pytest.fixture()
def file_loader(tmpdir):
    return FileLoader(cache_dir=tmpdir)


@pytest.mark.use_disk
def test_load_binary_task_csv(file_loader):
    ds_def = ns(
        train=os.path.join(res, "kc2_train.csv"),
        test=os.path.join(res, "kc2_test.csv"),
        target="problems"
    )
    ds = file_loader.load(ds_def)
    assert ds.type is DatasetType.binary
    assert len(ds.features) == 22
    assert len(ds.predictors) == 21

    assert ds.target.name == ds_def.target
    assert len(ds.target.values) == 2
    assert ds.target.values == ["no", "yes"]
    assert ds.target.data_type == 'string'   # file loader doesn't support categories yet when loading CSV files
    assert ds.target.is_target
    assert not ds.target.has_missing_values

    assert all([p.data_type in ['int', 'float'] for p in ds.predictors])
    assert all([p.values is None for p in ds.predictors])
    assert not any([p.is_target for p in ds.predictors])
    assert not any([p.has_missing_values for p in ds.predictors])

    assert ds.train.path == ds_def.train
    assert ds.test.path == ds_def.test
    assert ds.train.data_path('csv') == ds.train.path
    try:
        ds.train.data_path('arff')       # file loader doesn't support file auto-conversion yet
        pytest.fail("should have raised")
    except ValueError as e:
        assert "Dataset kc2_train is only available in csv format" in str(e)

    assert isinstance(ds.train.X, pd.DataFrame)
    assert isinstance(ds.train.y, pd.DataFrame)
    assert isinstance(ds.train.X_enc, np.ndarray)
    assert isinstance(ds.train.y_enc, np.ndarray)

    assert len(ds.train.X) == len(ds.train.y)
    assert len(ds.train.X.columns) == len(ds.predictors)
    assert len(ds.train.y.columns) == 1
    assert ds.train.y.columns == [ds.target.name]
    assert len(ds.train.X) > len(ds.test.X)
    floats = [p.name for p in ds.predictors if p.data_type == 'float']
    ints = [p.name for p in ds.predictors if p.data_type == 'int']
    assert ds.train.X.dtypes.filter(items=floats).apply(lambda dt: pd.api.types.is_float_dtype(dt)).all()
    assert ds.train.X.dtypes.filter(items=ints).apply(lambda dt: pd.api.types.is_integer_dtype(dt)).all()
    assert pd.api.types.is_string_dtype(ds.train.y.dtypes.iloc[0])
    assert ds.test.X.dtypes.equals(ds.train.X.dtypes)
    assert ds.test.y.dtypes.equals(ds.train.y.dtypes)

    assert ds.train.X_enc.shape == ds.train.X.shape
    assert np.issubdtype(ds.train.X_enc.dtype, np.floating)
    assert np.issubdtype(ds.train.y_enc.dtype, np.floating)  # not ideal given that it's binary, but well…


@pytest.mark.use_disk
def test_load_binary_task_arff(file_loader):
    ds_def = ns(
        train=os.path.join(res, "kc2_train.arff"),
        test=os.path.join(res, "kc2_test.arff"),
        target="problems"
    )
    ds = file_loader.load(ds_def)
    assert ds.type is DatasetType.binary
    assert len(ds.features) == 22
    assert len(ds.predictors) == 21

    assert ds.target.name == ds_def.target
    assert len(ds.target.values) == 2
    assert ds.target.values == ["no", "yes"]
    assert ds.target.data_type == 'category'
    assert ds.target.is_target
    assert not ds.target.has_missing_values

    assert all([p.data_type in ['int', 'float'] for p in ds.predictors])
    assert all([p.values is None for p in ds.predictors])
    assert not any([p.is_target for p in ds.predictors])
    assert not any([p.has_missing_values for p in ds.predictors])

    assert ds.train.path == ds_def.train
    assert ds.test.path == ds_def.test
    assert ds.train.data_path('arff') == ds.train.path
    try:
        ds.train.data_path('csv')       # file loader doesn't support file auto-conversion yet
        pytest.fail("should have raised")
    except ValueError as e:
        assert "Dataset kc2_train is only available in arff format" in str(e)

    assert isinstance(ds.train.X, pd.DataFrame)
    assert isinstance(ds.train.y, pd.DataFrame)
    assert isinstance(ds.train.X_enc, np.ndarray)
    assert isinstance(ds.train.y_enc, np.ndarray)

    assert len(ds.train.X) == len(ds.train.y)
    assert len(ds.train.X.columns) == len(ds.predictors)
    assert len(ds.train.y.columns) == 1
    assert ds.train.y.columns == [ds.target.name]
    assert len(ds.train.X) > len(ds.test.X)
    floats = [p.name for p in ds.predictors if p.data_type == 'float']
    ints = [p.name for p in ds.predictors if p.data_type == 'int']
    assert ds.train.X.dtypes.filter(items=floats).apply(lambda dt: pd.api.types.is_float_dtype(dt)).all()
    assert ds.train.X.dtypes.filter(items=ints).apply(lambda dt: pd.api.types.is_integer_dtype(dt)).all()
    assert pd.api.types.is_object_dtype(ds.train.y.dtypes.iloc[0])   # file loader doesn't represent categoricals as category dtype in pandas DF yet.
    assert ds.test.X.dtypes.equals(ds.train.X.dtypes)
    assert ds.test.y.dtypes.equals(ds.train.y.dtypes)


@pytest.mark.use_disk
def test_load_multiclass_task_csv(file_loader):
    ds_def = ns(
        train=os.path.join(res, "iris_train.csv"),
        test=os.path.join(res, "iris_test.csv"),
        target="class"
    )
    ds = file_loader.load(ds_def)
    assert ds.type is DatasetType.multiclass
    assert len(ds.features) == 5
    assert len(ds.predictors) == 4

    assert ds.target.name == ds_def.target
    assert len(ds.target.values) == 3
    assert ds.target.values == ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]  # values are case-sensitive when using file loader
    assert ds.target.data_type == 'string'   # file loader doesn't support categories yet when loading CSV files
    assert ds.target.is_target
    assert not ds.target.has_missing_values

    assert all([p.data_type in ['int', 'float'] for p in ds.predictors])
    assert all([p.values is None for p in ds.predictors])
    assert not any([p.is_target for p in ds.predictors])
    assert not any([p.has_missing_values for p in ds.predictors])

    assert ds.train.path == ds_def.train
    assert ds.test.path == ds_def.test
    assert ds.train.data_path('csv') == ds.train.path
    try:
        ds.train.data_path('arff')       # file loader doesn't support file auto-conversion yet
        pytest.fail("should have raised")
    except ValueError as e:
        assert "Dataset iris_train is only available in csv format" in str(e)

    assert isinstance(ds.train.X, pd.DataFrame)
    assert isinstance(ds.train.y, pd.DataFrame)
    assert isinstance(ds.train.X_enc, np.ndarray)
    assert isinstance(ds.train.y_enc, np.ndarray)

    assert len(ds.train.X) == len(ds.train.y)
    assert len(ds.train.X.columns) == len(ds.predictors)
    assert len(ds.train.y.columns) == 1
    assert ds.train.y.columns == [ds.target.name]
    assert len(ds.train.X) > len(ds.test.X)
    floats = [p.name for p in ds.predictors if p.data_type == 'float']
    ints = [p.name for p in ds.predictors if p.data_type == 'int']
    assert ds.train.X.dtypes.filter(items=floats).apply(lambda dt: pd.api.types.is_float_dtype(dt)).all()
    assert ds.train.X.dtypes.filter(items=ints).apply(lambda dt: pd.api.types.is_integer_dtype(dt)).all()
    assert pd.api.types.is_string_dtype(ds.train.y.dtypes.iloc[0])
    assert ds.test.X.dtypes.equals(ds.train.X.dtypes)
    assert ds.test.y.dtypes.equals(ds.train.y.dtypes)


@pytest.mark.use_disk
def test_load_multiclass_task_arff(file_loader):
    ds_def = ns(
        train=os.path.join(res, "iris_train.arff"),
        test=os.path.join(res, "iris_test.arff"),
        target="class"
    )
    ds = file_loader.load(ds_def)
    assert ds.type is DatasetType.multiclass
    assert len(ds.features) == 5
    assert len(ds.predictors) == 4

    assert ds.target.name == ds_def.target
    assert len(ds.target.values) == 3
    assert ds.target.values == ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]  # values are case-sensitive when using file loader
    assert ds.target.data_type == 'category'
    assert ds.target.is_target
    assert not ds.target.has_missing_values

    assert all([p.data_type in ['int', 'float'] for p in ds.predictors])
    assert all([p.values is None for p in ds.predictors])
    assert not any([p.is_target for p in ds.predictors])
    assert not any([p.has_missing_values for p in ds.predictors])

    assert ds.train.path == ds_def.train
    assert ds.test.path == ds_def.test
    assert ds.train.data_path('arff') == ds.train.path
    try:
        ds.train.data_path('csv')       # file loader doesn't support file auto-conversion yet
        pytest.fail("should have raised")
    except ValueError as e:
        assert "Dataset iris_train is only available in arff format" in str(e)

    assert isinstance(ds.train.X, pd.DataFrame)
    assert isinstance(ds.train.y, pd.DataFrame)
    assert isinstance(ds.train.X_enc, np.ndarray)
    assert isinstance(ds.train.y_enc, np.ndarray)

    assert len(ds.train.X) == len(ds.train.y)
    assert len(ds.train.X.columns) == len(ds.predictors)
    assert len(ds.train.y.columns) == 1
    assert ds.train.y.columns == [ds.target.name]
    assert len(ds.train.X) > len(ds.test.X)
    floats = [p.name for p in ds.predictors if p.data_type == 'float']
    ints = [p.name for p in ds.predictors if p.data_type == 'int']
    assert ds.train.X.dtypes.filter(items=floats).apply(lambda dt: pd.api.types.is_float_dtype(dt)).all()
    assert ds.train.X.dtypes.filter(items=ints).apply(lambda dt: pd.api.types.is_integer_dtype(dt)).all()
    assert pd.api.types.is_object_dtype(ds.train.y.dtypes.iloc[0])   # file loader doesn't represent categoricals as category dtype in pandas DF yet.
    assert ds.test.X.dtypes.equals(ds.train.X.dtypes)
    assert ds.test.y.dtypes.equals(ds.train.y.dtypes)


@pytest.mark.use_disk
def test_load_regression_task_csv(file_loader):
    ds_def = ns(
        train=os.path.join(res, "cholesterol_train.csv"),
        test=os.path.join(res, "cholesterol_test.csv"),
        target="chol"
    )
    ds = file_loader.load(ds_def)
    assert ds.type is DatasetType.regression
    assert len(ds.features) == 14
    assert len(ds.predictors) == 13

    assert ds.target.name == ds_def.target
    assert ds.target.values is None
    assert ds.target.data_type == 'float'
    assert ds.target.is_target
    assert not ds.target.has_missing_values

    assert len([p for p in ds.predictors if p.data_type == 'int']) == 6  # types are detected differently in CSV format
    assert len([p for p in ds.predictors if p.data_type == 'float']) == 7
    assert not any([p.is_target for p in ds.predictors])
    assert len([p for p in ds.predictors if p.has_missing_values]) == 2

    assert ds.train.path == ds_def.train
    assert ds.test.path == ds_def.test
    assert ds.train.data_path('csv') == ds.train.path
    try:
        ds.train.data_path('arff')       # file loader doesn't support file auto-conversion yet
        pytest.fail("should have raised")
    except ValueError as e:
        assert "Dataset cholesterol_train is only available in csv format" in str(e)

    assert isinstance(ds.train.X, pd.DataFrame)
    assert isinstance(ds.train.y, pd.DataFrame)
    assert isinstance(ds.train.X_enc, np.ndarray)
    assert isinstance(ds.train.y_enc, np.ndarray)

    assert len(ds.train.X) == len(ds.train.y)
    assert len(ds.train.X.columns) == len(ds.predictors)
    assert len(ds.train.y.columns) == 1
    assert ds.train.y.columns == [ds.target.name]
    assert len(ds.train.X) > len(ds.test.X)
    ints = [p.name for p in ds.predictors if p.data_type == 'int']
    floats = [p.name for p in ds.predictors if p.data_type == 'float']
    assert ds.train.X.dtypes.filter(items=ints).apply(lambda dt: pd.api.types.is_integer_dtype(dt)).all()
    assert ds.train.X.dtypes.filter(items=floats).apply(lambda dt: pd.api.types.is_float_dtype(dt)).all()
    assert pd.api.types.is_float_dtype(ds.train.y.dtypes.iloc[0])
    assert ds.test.X.dtypes.equals(ds.train.X.dtypes)
    assert ds.test.y.dtypes.equals(ds.train.y.dtypes)


@pytest.mark.use_disk
def test_load_regression_task_arff(file_loader):
    ds_def = ns(
        train=os.path.join(res, "cholesterol_train.arff"),
        test=os.path.join(res, "cholesterol_test.arff"),
        target="chol"
    )
    ds = file_loader.load(ds_def)
    assert ds.type is DatasetType.regression
    assert len(ds.features) == 14
    assert len(ds.predictors) == 13

    assert ds.target.name == ds_def.target
    assert ds.target.values is None
    assert ds.target.data_type == 'float'
    assert ds.target.is_target
    assert not ds.target.has_missing_values

    assert len([p for p in ds.predictors if p.data_type == 'float']) == 6
    assert len([p for p in ds.predictors if p.data_type == 'category']) == 7
    assert not any([p.is_target for p in ds.predictors])
    assert len([p for p in ds.predictors if p.has_missing_values]) == 2

    assert ds.train.path == ds_def.train
    assert ds.test.path == ds_def.test
    assert ds.train.data_path('arff') == ds.train.path
    try:
        ds.train.data_path('csv')       # file loader doesn't support file auto-conversion yet
        pytest.fail("should have raised")
    except ValueError as e:
        assert "Dataset cholesterol_train is only available in arff format" in str(e)

    assert isinstance(ds.train.X, pd.DataFrame)
    assert isinstance(ds.train.y, pd.DataFrame)
    assert isinstance(ds.train.X_enc, np.ndarray)
    assert isinstance(ds.train.y_enc, np.ndarray)

    assert len(ds.train.X) == len(ds.train.y)
    assert len(ds.train.X.columns) == len(ds.predictors)
    assert len(ds.train.y.columns) == 1
    assert ds.train.y.columns == [ds.target.name]
    assert len(ds.train.X) > len(ds.test.X)
    categoricals = [p.name for p in ds.predictors if p.data_type == 'category']
    floats = [p.name for p in ds.predictors if p.data_type == 'float']
    assert ds.train.X.dtypes.filter(items=categoricals).apply(lambda dt: pd.api.types.is_object_dtype(dt)).all()
    assert ds.train.X.dtypes.filter(items=floats).apply(lambda dt: pd.api.types.is_float_dtype(dt)).all()
    assert pd.api.types.is_float_dtype(ds.train.y.dtypes.iloc[0])
    assert ds.test.X.dtypes.equals(ds.train.X.dtypes)
    assert ds.test.y.dtypes.equals(ds.train.y.dtypes)

