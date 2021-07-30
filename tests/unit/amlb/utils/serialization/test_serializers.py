import os

import pytest

from amlb.utils.core import Namespace as ns
from amlb.utils.serialization import is_sparse, serialize_data, deserialize_data


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
    ser = pd.Series([1, 2.2, pd.NA, 3, 4.4])
    dest = os.path.join(tmpdir, "my_pd_ser")
    path = serialize_data(ser, dest)
    assert path == f"{dest}.pd"

    reloaded = deserialize_data(path)
    assert isinstance(reloaded, pd.Series)
    assert ser.compare(reloaded).empty


@pytest.mark.use_disk
def test_serialize_pandas_dataframes(tmpdir):
    import pandas as pd
    df = pd.DataFrame(dict(
        first=[1, 2.2, pd.NA, 3, 4.4],
        second=['a', 'b', 'c', 'a', 'b']
    ))
    dest = os.path.join(tmpdir, "my_pd_df")
    path = serialize_data(df, dest)
    assert path == f"{dest}.pd"

    reloaded = deserialize_data(path)
    assert isinstance(reloaded, pd.DataFrame)
    assert df.compare(reloaded).empty


@pytest.mark.use_disk
def test_serialize_sparse_matrix(tmpdir):
    import scipy.sparse as sp
    import numpy as np
    arr = np.array([[0, 0, 0, 3.3], [4.4, 0, 0, 0], [0, np.nan, 0, 0]])
    nans = np.count_nonzero(np.isnan(arr))
    mat = sp.csc_matrix(arr)
    assert sp.issparse(mat)
    dest = os.path.join(tmpdir, "my_sparse_mat")
    path = serialize_data(mat, dest)
    assert path == f"{dest}.spy.npz"

    reloaded = deserialize_data(path, config=ns(sparse_matrix_deserialized_format=None))
    assert isinstance(reloaded, sp.spmatrix)
    assert (mat != reloaded).nnz == nans
    assert np.array_equal(mat.toarray(), reloaded.toarray(), equal_nan=True)


@pytest.mark.use_disk
def test_serialize_sparse_matrix_reload_as_dense(tmpdir):
    import scipy.sparse as sp
    import numpy as np
    arr = np.array([[0, 0, 0, 3.3], [4.4, 0, 0, 0], [0, np.nan, 0, 0]])
    mat = sp.csc_matrix(arr)
    assert sp.issparse(mat)
    dest = os.path.join(tmpdir, "my_sparse_mat")
    path = serialize_data(mat, dest)
    assert path == f"{dest}.spy.npz"

    reloaded = deserialize_data(path, config=ns(sparse_matrix_deserialized_format='dense'))
    assert not sp.issparse(reloaded)
    assert isinstance(reloaded, np.matrix)
    assert np.array_equal(mat.toarray(), np.asarray(reloaded), equal_nan=True)


@pytest.mark.use_disk
def test_serialize_sparse_matrix_reload_as_array(tmpdir):
    import scipy.sparse as sp
    import numpy as np
    arr = np.array([[0, 0, 0, 3.3], [4.4, 0, 0, 0], [0, np.nan, 0, 0]])
    mat = sp.csc_matrix(arr)
    assert sp.issparse(mat)
    dest = os.path.join(tmpdir, "my_sparse_mat")
    path = serialize_data(mat, dest)
    assert path == f"{dest}.spy.npz"

    reloaded = deserialize_data(path, config=ns(sparse_matrix_deserialized_format='array'))
    assert isinstance(reloaded, np.ndarray)
    assert np.array_equal(mat.toarray(), reloaded, equal_nan=True)


@pytest.mark.use_disk
def test_serialize_sparse_dataframe(tmpdir):
    import pandas as pd
    ser_config = ns(pandas_serializer='pickle', sparse_dataframe_deserialized_format=None)
    dfs = pd.DataFrame(dict(
        first=[0, 0, 0, 3.3],
        second=[4.4, 0, 0, 0],
        third=[0, pd.NA, 0, 0],
    )).astype('Sparse')
    assert is_sparse(dfs)
    dest = os.path.join(tmpdir, "my_sparse_df")
    path = serialize_data(dfs, dest, config=ser_config)
    assert path == f"{dest}.pd"

    reloaded = deserialize_data(path, config=ser_config)
    assert isinstance(reloaded, pd.DataFrame)
    assert is_sparse(reloaded)
    assert dfs.compare(reloaded).empty


@pytest.mark.use_disk
def test_serialize_pandas_dataframe_reload_as_dense(tmpdir):
    import pandas as pd
    ser_config = ns(pandas_serializer='pickle', sparse_dataframe_deserialized_format='dense')
    dfs = pd.DataFrame(dict(
        first=[0, 0, 0, 3.3],
        second=[4.4, 0, 0, 0],
        third=[0, pd.NA, 0, 0],
        # fourth=[None, None, 'a', None]
    )).astype('Sparse')
    assert is_sparse(dfs)
    dest = os.path.join(tmpdir, "my_sparse_df")
    path = serialize_data(dfs, dest, config=ser_config)
    assert path == f"{dest}.pd"

    reloaded = deserialize_data(path, config=ser_config)
    assert isinstance(reloaded, pd.DataFrame)
    assert not is_sparse(reloaded)
    assert dfs.compare(reloaded).empty


@pytest.mark.use_disk
def test_serialize_pandas_dataframe_reload_as_array(tmpdir):
    import numpy as np
    import pandas as pd
    ser_config = ns(pandas_serializer='pickle', sparse_dataframe_deserialized_format='array')
    dfs = pd.DataFrame(dict(
        first=[0, 0, 0, 3.3],
        second=[4.4, 0, 0, 0],
        third=[0, pd.NA, 0, 0],
        # fourth=[None, None, 'a', None]
    )).astype('Sparse')
    assert is_sparse(dfs)
    dest = os.path.join(tmpdir, "my_sparse_df")
    path = serialize_data(dfs, dest, config=ser_config)
    assert path == f"{dest}.pd"

    reloaded = deserialize_data(path, config=ser_config)
    assert isinstance(reloaded, np.ndarray)
    assert np.array_equal(dfs.to_numpy(), np.asarray(reloaded), equal_nan=True)


@pytest.mark.use_disk
def test_serialize_sparse_numerical_dataframe_to_parquet(tmpdir):
    import pandas as pd
    ser_config = ns(pandas_serializer='parquet', sparse_dataframe_deserialized_format=None)
    dfs = pd.DataFrame(dict(
        first=[0, 0, 0, 3.3],
        second=[4.4, 0, 0, 0],
        third=[0, pd.NA, 0, 0],
    )).astype('Sparse')
    assert is_sparse(dfs)
    dest = os.path.join(tmpdir, "my_sparse_df")
    path = serialize_data(dfs, dest, config=ser_config)
    assert path == f"{dest}.sparse.pd"

    reloaded = deserialize_data(path, config=ser_config)
    assert isinstance(reloaded, pd.DataFrame)
    assert is_sparse(reloaded)
    assert dfs.compare(reloaded).empty


@pytest.mark.use_disk
def test_serialize_mixed_dataframe_to_parquet(tmpdir):
    import pandas as pd
    ser_config = ns(pandas_serializer='parquet', sparse_dataframe_deserialized_format=None)
    dfm = pd.DataFrame(dict(
        first=pd.arrays.SparseArray([0, 0, 0, 3.3]),
        second=pd.arrays.SparseArray([4.4, 0, 0, 0], dtype=pd.SparseDtype(float, 0)),
        third=pd.arrays.SparseArray([0, pd.NA, 0, 0]),
        fourth=[None, None, 'a', None]
    ))
    assert is_sparse(dfm)
    dest = os.path.join(tmpdir, "my_mixed_df")
    path = serialize_data(dfm, dest, config=ser_config)
    assert path == f"{dest}.sparse.pd"

    reloaded = deserialize_data(path, config=ser_config)
    assert isinstance(reloaded, pd.DataFrame)
    assert is_sparse(reloaded)
    assert dfm.compare(reloaded).empty


