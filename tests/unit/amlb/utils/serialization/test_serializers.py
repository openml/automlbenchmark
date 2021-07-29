import os

import pytest

from amlb.utils.serialization import serialize_data, deserialize_data


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


# more to come
