import copy
import os

import numpy as np
import pandas as pd
import pytest
import pandas.api.types as pat

from amlb.resources import from_config
from amlb.data import DatasetType
from amlb.datasets.file import FileLoader
from amlb.utils import Namespace as ns, path_from_split, split_path

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
    _assert_X_y_types(ds.train)
    _assert_data_consistency(ds)
    _assert_data_paths(ds, ds_def)
    _assert_kc2_features(ds, ds_def)



@pytest.mark.use_disk
def test_load_binary_task_arff(file_loader):
    ds_def = ns(
        train=os.path.join(res, "kc2_train.arff"),
        test=os.path.join(res, "kc2_test.arff"),
        target="problems"
    )
    ds = file_loader.load(ds_def)
    assert ds.type is DatasetType.binary
    _assert_X_y_types(ds.train)
    _assert_data_consistency(ds)
    _assert_data_paths(ds, ds_def)
    _assert_kc2_features(ds, ds_def)


def _assert_kc2_features(dataset, definition):
    assert len(dataset.features) == 22
    assert len(dataset.predictors) == 21
    target_values = ["no", "yes"]
    _assert_target(dataset.target, name=definition.target, values=target_values)

    assert all([p.data_type in ['int', 'float'] for p in dataset.predictors])
    assert all([p.values is None for p in dataset.predictors])
    assert not any([p.is_target for p in dataset.predictors])
    assert not any([p.has_missing_values for p in dataset.predictors])

    floats = [p.name for p in dataset.predictors if p.data_type == 'float']
    ints = [p.name for p in dataset.predictors if p.data_type == 'int']
    assert dataset.train.X.dtypes.filter(items=floats).apply(lambda dt: pd.api.types.is_float_dtype(dt)).all()
    assert dataset.train.X.dtypes.filter(items=ints).apply(lambda dt: pd.api.types.is_integer_dtype(dt)).all()
    assert pd.api.types.is_categorical_dtype(dataset.train.y.dtypes.iloc[0])

    normalize = dataset.target.normalize
    assert list(normalize(dataset.train.y.squeeze().unique())) == list(normalize(dataset.test.y.squeeze().unique())) == target_values
    assert list(np.unique(dataset.train.y_enc)) == list(np.unique(dataset.test.y_enc)) == [0, 1]


@pytest.mark.use_disk
def test_load_multiclass_task_csv(file_loader):
    ds_def = ns(
        train=os.path.join(res, "iris_train.csv"),
        test=os.path.join(res, "iris_test.csv"),
        target="class"
    )
    ds = file_loader.load(ds_def)
    assert ds.type is DatasetType.multiclass
    _assert_X_y_types(ds.train)
    _assert_data_consistency(ds)
    _assert_data_paths(ds, ds_def)
    _assert_iris_features(ds, ds_def)


@pytest.mark.use_disk
def test_load_multiclass_task_with_num_target_no_type_csv(file_loader):
    ds_def = ns(
        train=os.path.join(res, "iris_num_train.csv"),
        test=os.path.join(res, "iris_num_test.csv"),
        target="class"
    )
    ds = file_loader.load(ds_def)
    assert ds.type is DatasetType.regression, "file loader should detect num target as regression by default"


@pytest.mark.use_disk
def test_load_multiclass_task_with_num_target_csv(file_loader):
    ds_def = ns(
        train=os.path.join(res, "iris_num_train.csv"),
        test=os.path.join(res, "iris_num_test.csv"),
        target="class",
        type="multiclass"
    )
    ds = file_loader.load(ds_def)
    assert ds.type is DatasetType.multiclass
    _assert_X_y_types(ds.train)
    _assert_data_consistency(ds)
    _assert_data_paths(ds, ds_def)
    _assert_iris_features(ds, ds_def, num_target=True)


@pytest.mark.use_disk
def test_load_multiclass_task_arff(file_loader):
    ds_def = ns(
        train=os.path.join(res, "iris_train.arff"),
        test=os.path.join(res, "iris_test.arff"),
        target="class"
    )
    ds = file_loader.load(ds_def)
    assert ds.type is DatasetType.multiclass
    _assert_X_y_types(ds.train)
    _assert_data_consistency(ds)
    _assert_data_paths(ds, ds_def)
    _assert_iris_features(ds, ds_def)


def _assert_iris_features(dataset, definition, num_target=False):
    assert len(dataset.features) == 5
    assert len(dataset.predictors) == 4
    target_values = ["1", "2", "3"] if num_target else ["iris-setosa", "iris-versicolor", "iris-virginica"]  # values are normalized
    _assert_target(dataset.target, name=definition.target, values=target_values)

    assert all([p.data_type in ['int', 'float'] for p in dataset.predictors])
    assert all([p.values is None for p in dataset.predictors])
    assert not any([p.is_target for p in dataset.predictors])
    assert not any([p.has_missing_values for p in dataset.predictors])

    floats = [p.name for p in dataset.predictors if p.data_type == 'float']
    ints = [p.name for p in dataset.predictors if p.data_type == 'int']
    assert dataset.train.X.dtypes.filter(items=floats).apply(lambda dt: pd.api.types.is_float_dtype(dt)).all()
    assert dataset.train.X.dtypes.filter(items=ints).apply(lambda dt: pd.api.types.is_integer_dtype(dt)).all()
    assert pd.api.types.is_categorical_dtype(dataset.train.y.dtypes.iloc[0])

    normalize = dataset.target.normalize
    assert list(normalize(dataset.train.y.squeeze().unique())) == list(normalize(dataset.test.y.squeeze().unique())) == target_values
    assert list(np.unique(dataset.train.y_enc)) == list(np.unique(dataset.test.y_enc)) == [0, 1, 2]


@pytest.mark.use_disk
def test_load_regression_task_csv(file_loader):
    ds_def = ns(
        train=os.path.join(res, "cholesterol_train.csv"),
        test=os.path.join(res, "cholesterol_test.csv"),
        target="chol"
    )
    ds = file_loader.load(ds_def)
    assert ds.type is DatasetType.regression
    print(ds.train.X.dtypes)
    _assert_X_y_types(ds.train)
    _assert_data_consistency(ds)
    _assert_data_paths(ds, ds_def)
    _assert_cholesterol_features(ds, ds_def, 'csv')


@pytest.mark.use_disk
def test_load_regression_task_arff(file_loader):
    ds_def = ns(
        train=os.path.join(res, "cholesterol_train.arff"),
        test=os.path.join(res, "cholesterol_test.arff"),
        target="chol"
    )
    ds = file_loader.load(ds_def)
    assert ds.type is DatasetType.regression
    print(ds.train.X.dtypes)
    _assert_X_y_types(ds.train)
    _assert_data_consistency(ds)
    _assert_data_paths(ds, ds_def)
    _assert_cholesterol_features(ds, ds_def, 'arff')


def _assert_cholesterol_features(dataset, definition, fmt):
    assert len(dataset.features) == 14
    assert len(dataset.predictors) == 13
    _assert_target(dataset.target, name=definition.target)

    ints = [p.name for p in dataset.predictors if p.data_type == 'int']
    floats = [p.name for p in dataset.predictors if p.data_type == 'float']
    categoricals = [p.name for p in dataset.predictors if p.data_type == 'category']

    assert len(ints) == (0 if fmt == 'arff' else 6)
    assert len(floats) == (6 if fmt == 'arff' else 7)
    assert len(categoricals) == (7 if fmt == 'arff' else 0)
    assert not any([p.is_target for p in dataset.predictors])
    assert len([p for p in dataset.predictors if p.has_missing_values]) == 2

    assert dataset.train.X.dtypes.filter(items=ints).apply(lambda dt: pd.api.types.is_integer_dtype(dt)).all()
    assert dataset.train.X.dtypes.filter(items=floats).apply(lambda dt: pd.api.types.is_float_dtype(dt)).all()
    assert dataset.train.X.dtypes.filter(items=categoricals).apply(lambda dt: pd.api.types.is_categorical_dtype(dt)).all()
    assert pd.api.types.is_float_dtype(dataset.train.y.dtypes.iloc[0])

    assert np.array_equal(dataset.train.y_enc, dataset.train.y.squeeze().to_numpy()), "no encoding should have been applied on regression target"
    assert np.array_equal(dataset.test.y_enc, dataset.test.y.squeeze().to_numpy()), "no encoding should have been applied on regression target"


def _assert_target(target, name, values=None):
    assert target.name == name
    assert target.values == values
    assert target.data_type == 'category' if values else 'float'
    assert target.is_target
    assert not target.has_missing_values


def _assert_data_paths(dataset, definition):
    assert dataset.train.path == definition.train
    assert dataset.test.path == definition.test
    sp = split_path(definition.train)
    fmt = sp.extension[1:]
    for f in ['arff', 'csv', 'parquet']:
        if f == fmt:
            assert dataset.train.data_path(f) == dataset.train.path
        else:
            s = copy.copy(sp)
            s.extension = f
            assert dataset.train.data_path(f) == path_from_split(s)


def _assert_X_y_types(data_split, check_encoded=True):
    assert isinstance(data_split.X, pd.DataFrame)
    assert isinstance(data_split.y, pd.DataFrame)
    if check_encoded:
        assert isinstance(data_split.X_enc, np.ndarray)
        assert isinstance(data_split.y_enc, np.ndarray)


def _assert_data_consistency(dataset, check_encoded=True):
    assert len(dataset.train.X) == len(dataset.train.y)
    assert len(dataset.train.X.columns) == len(dataset.predictors)
    assert len(dataset.train.y.columns) == 1
    assert dataset.train.y.columns == [dataset.target.name]
    assert len(dataset.train.X) > len(dataset.test.X)

    assert not any([p.is_target for p in dataset.predictors])


    assert dataset.test.X.dtypes.equals(dataset.train.X.dtypes)
    assert dataset.test.y.dtypes.equals(dataset.train.y.dtypes)

    if check_encoded:
        assert dataset.train.X_enc.shape == dataset.train.X.shape
        assert np.issubdtype(dataset.train.X_enc.dtype, np.floating)
        assert np.issubdtype(dataset.train.y_enc.dtype, np.floating)  # not ideal given that it's also for classification targets, but well…



@pytest.mark.use_disk
def test_load_timeseries_task_csv(file_loader):
    ds_def = ns(
        path=os.path.join(res, "m4_hourly_subset.csv"),
        forecast_horizon_in_steps=24,
        seasonality=24,
        freq="H",
        target="target",
        type="timeseries",
    )
    ds = file_loader.load(ds_def)
    assert ds.type is DatasetType.timeseries
    print(ds.train.X.dtypes)
    _assert_data_consistency(ds, check_encoded=False)
    _assert_X_y_types(ds.train, check_encoded=False)
    _assert_X_y_types(ds.test, check_encoded=False)

    assert isinstance(ds.train.data, pd.DataFrame)
    assert isinstance(ds.test.data, pd.DataFrame)
    assert len(ds.repeated_abs_seasonal_error) == len(ds.test.data)
    assert len(ds.repeated_item_id) == len(ds.test.data)

    assert pat.is_string_dtype(ds._dtypes[ds.id_column])
    assert pat.is_datetime64_dtype(ds._dtypes[ds.timestamp_column])
    assert pat.is_float_dtype(ds._dtypes[ds.target.name])

    # timeseries uses different task schema - set attributes for test to work
    ds_def['train'] = ds.train.path
    ds_def['test'] = ds.test.path
    _assert_data_paths(ds, ds_def)


@pytest.mark.parametrize("missing_key", ["freq", "forecast_horizon_in_steps", "seasonality"])
def test_when_timeseries_task_key_is_missing_then_exception_is_raised(file_loader, missing_key):
    task_kwargs = dict(
        path=os.path.join(res, "m4_hourly_subset.csv"),
        forecast_horizon_in_steps=24,
        seasonality=24,
        freq="H",
        target="target",
        type="timeseries",
    )
    task_kwargs.pop(missing_key)
    ds_def = ns.from_dict(task_kwargs)
    with pytest.raises(AssertionError, match=f"Task definition for timeseries must include `{missing_key}`"):
        file_loader.load(ds_def)


@pytest.mark.parametrize("missing_key", ["id_column", "timestamp_column"])
def test_given_nondefault_column_names_when_key_is_missing_then_exception_is_raised(file_loader, missing_key):
    task_kwargs = dict(
        path=os.path.join(res, "m4_hourly_subset_nondefault_cols.csv"),
        forecast_horizon_in_steps=24,
        seasonality=24,
        freq="H",
        type="timeseries",
        target="CustomTarget",
        id_column="CustomId",
        timestamp_column="CustomTimestamp",
    )
    task_kwargs.pop(missing_key)
    ds_def = ns.from_dict(task_kwargs)
    with pytest.raises(ValueError, match=missing_key):
        file_loader.load(ds_def)


def test_given_nondefault_column_names_then_timeseries_dataset_can_be_loaded(file_loader):
    task_kwargs = dict(
        path=os.path.join(res, "m4_hourly_subset_nondefault_cols.csv"),
        forecast_horizon_in_steps=24,
        seasonality=24,
        freq="H",
        type="timeseries",
        target="CustomTarget",
        id_column="CustomId",
        timestamp_column="CustomTimestamp",
    )
    ds_def = ns.from_dict(task_kwargs)
    ds = file_loader.load(ds_def)
    _assert_data_consistency(ds, check_encoded=False)


@pytest.mark.parametrize("forecast_horizon, fold", [(50, 2), (100, 0), (10, 9)])
def test_if_timeseries_dataset_too_short_for_requested_fold_then_exception_is_raised(file_loader, forecast_horizon, fold):
    ds_def = ns(
        path=os.path.join(res, "m4_hourly_subset.csv"),
        forecast_horizon_in_steps=forecast_horizon,
        seasonality=24,
        freq="H",
        type="timeseries",
    )
    with pytest.raises(ValueError, match="All time series in the dataset must have length"):
        file_loader.load(ds_def, fold=fold)
