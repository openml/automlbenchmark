import os

import pytest

from amlb.utils.core import Namespace as ns
from amlb.utils.serialization import serialize_data, deserialize_data


@pytest.mark.use_disk
def test_serialize_list_json(tmpdir):
    li = [[1, 2.2, None, 3, 4.4, 'foo', True], ['bar', False, 2/3]]
    dest = os.path.join(tmpdir, "my_list")
    path = serialize_data(li, dest, config=ns(fallback_serializer='json'))
    assert path == f"{dest}.json"

    reloaded = deserialize_data(path)
    assert isinstance(reloaded, list)
    assert li == reloaded


@pytest.mark.use_disk
def test_serialize_list_pickle(tmpdir):
    li = [[1, 2.2, None, 3, 4.4, 'foo', True], ['bar', False, 2/3]]
    dest = os.path.join(tmpdir, "my_list")
    path = serialize_data(li, dest, config=ns(fallback_serializer='pickle'))
    assert path == f"{dest}.pkl"

    reloaded = deserialize_data(path)
    assert isinstance(reloaded, list)
    assert li == reloaded


@pytest.mark.use_disk
def test_serialize_dict_json(tmpdir):
    di = dict(
        first=[1, 2.2, None, 3, 4.4, 'foo', True],
        second=['bar', False, 2/3]
    )
    dest = os.path.join(tmpdir, "my_dict")
    path = serialize_data(di, dest, config=ns(fallback_serializer='json'))
    assert path == f"{dest}.json"

    reloaded = deserialize_data(path)
    assert isinstance(reloaded, dict)
    assert di == reloaded


@pytest.mark.use_disk
def test_serialize_dict_pickle(tmpdir):
    di = dict(
        first=[1, 2.2, None, 3, 4.4, 'foo', True],
        second=['bar', False, 2/3]
    )
    dest = os.path.join(tmpdir, "my_dict")
    path = serialize_data(di, dest, config=ns(fallback_serializer='pickle'))
    assert path == f"{dest}.pkl"

    reloaded = deserialize_data(path)
    assert isinstance(reloaded, dict)
    assert di == reloaded


@pytest.mark.use_disk
def test_serialize_numpy_array(tmpdir):
    import numpy as np
    arr = np.array([1, 2.2, np.nan, 3, 4.4])
    dest = os.path.join(tmpdir, "my_np_arr")
    path = serialize_data(arr, dest)
    assert path == f"{dest}.npy"

    reloaded = deserialize_data(path)
    assert isinstance(reloaded, np.ndarray)
    assert np.array_equal(arr, reloaded, equal_nan=True)


@pytest.mark.use_disk
def test_serialize_pandas_series(tmpdir):
    import pandas as pd
    arr = pd.Series([1, 2.2, pd.NA, 3, 4.4])
    dest = os.path.join(tmpdir, "my_pd_ser")
    path = serialize_data(arr, dest)
    assert path == f"{dest}.pd"

    reloaded = deserialize_data(path)
    assert isinstance(reloaded, pd.Series)
    assert arr.compare(reloaded).empty


@pytest.mark.use_disk
def test_serialize_pandas_dataframes(tmpdir):
    import pandas as pd
    arr = pd.DataFrame(dict(
        first=[1, 2.2, pd.NA, 3, 4.4],
        second=['a', 'b', 'c', 'a', 'b']
    ))
    dest = os.path.join(tmpdir, "my_pd_df")
    path = serialize_data(arr, dest)
    assert path == f"{dest}.pd"

    reloaded = deserialize_data(path)
    assert isinstance(reloaded, pd.DataFrame)
    assert arr.compare(reloaded).empty


@pytest.mark.use_disk
def test_serialize_sparse_matrix(tmpdir):
    pass


@pytest.mark.use_disk
def test_serialize_sparse_matrix_reload_as_dense(tmpdir):
    pass


@pytest.mark.use_disk
def test_serialize_sparse_matrix_reload_as_array(tmpdir):
    pass


@pytest.mark.use_disk
def test_serialize_sparse_dataframe(tmpdir):
    pass


@pytest.mark.use_disk
def test_serialize_pandas_dataframe_reload_as_dense(tmpdir):
    pass


@pytest.mark.use_disk
def test_serialize_pandas_dataframe_reload_as_array(tmpdir):
    pass


