import logging
import os

from .utils import Namespace as ns, profile

log = logging.getLogger(__name__)


def _import_data_libraries():
    try:
        import numpy as np
    except ImportError:
        np = None
    try:
        import pandas as pd
    except ImportError:
        pd = None
    try:
        import scipy.sparse as sp
    except ImportError:
        sp = None
    return np, pd, sp


ser_config = ns(
    # 'pickle', 'parquet', 'hdf', 'json'
    pandas_serializer=os.environ.get('AMLB_SER_PD_MODE') or 'parquet',
    # 'infer', 'bz2', 'gzip', None ?
    # 'infer' (here=None) is the fastest but no compression,
    # 'gzip' fast write and read with good compression.
    # 'bz2' looks like the best compression/time ratio (faster write, sometimes slightly slower read)
    pandas_compression=os.environ.get('AMLB_SER_PD_COMPR') or 'infer',
    # 'snappy', 'gzip', 'brotli', None ?
    pandas_parquet_compression=os.environ.get('AMLB_SER_PD_PQT_COMPR') or None,
    # if numpy can use pickle to serialize ndarrays,
    numpy_pickle=True,
    # if sparse matrices should be compressed during serialization.
    sparse_matrix_compression=True,
    # if sparse matrices should be deserialized to some specific format:
    #  None (no change), 'array' (numpy), 'dense' (dense matrix).
    sparse_matrix_deserialized_format='array',
    # if sparse dataframes should be deserialized to some specific format:
    #  None (no change), 'array' (numpy), 'dense' (dense dataframe/series).
    sparse_dataframe_deserialized_format=None
)


__series__ = '_series_'


def is_serializable_data(data):
    np, pd, sp = _import_data_libraries()
    return isinstance(data, (np.ndarray, sp.spmatrix, pd.DataFrame, pd.Series))


def unsparsify(*data, fmt=None):
    if len(data) == 1:
        return _unsparsify(data[0], fmt=fmt)
    else:
        return tuple(_unsparsify(d, fmt=fmt) for d in data)


def _unsparsify(data, fmt=None):
    """
    :param data: the matrix to process.
    :param fmt: one of None, 'array', 'dense'
    :return: the original matrix is fmt is None,
            a numpy array if fmt is 'array',
            a dense version of the data type if fmt is 'dense'.
    """
    if fmt is None:
        return data
    np, pd, sp = _import_data_libraries()
    if sp and isinstance(data, sp.spmatrix):
        return (data.toarray() if fmt == 'array'
                else data.todense() if fmt == 'dense'
                else data)
    elif pd and isinstance(data, (pd.DataFrame, pd.Series)):
        return (data.to_numpy() if fmt == 'array'
                else data.sparse.to_dense() if fmt == 'dense' and hasattr(data, 'sparse')
                else data)
    else:
        return data


@profile(log)
def serialize_data(data, path):
    np, pd, sp = _import_data_libraries()
    if np and isinstance(data, np.ndarray):
        root, ext = os.path.splitext(path)
        path = f"{root}.npy"
        np.save(path, data, allow_pickle=ser_config.numpy_pickle)
    elif sp and isinstance(data, sp.spmatrix):
        root, ext = os.path.splitext(path)
        # use custom extension to recognize sparsed matrices from file name.
        # .npz is automatically appended if missing, and can also potentially be used for numpy arrays.
        path = f"{root}.spy.npz"
        sp.save_npz(path, data, compressed=ser_config.sparse_matrix_compression)
    elif pd and isinstance(data, (pd.DataFrame, pd.Series)):
        if isinstance(data, pd.DataFrame):
            # pandas has this habit of inferring value types when data are loaded from file,
            # for example, 'true' and 'false' are converted automatically to booleans, even for column names…
            data.rename(str, axis='columns', inplace=True)
        ser = ser_config.pandas_serializer
        if ser == 'pickle':
            data.to_pickle(path, compression=ser_config.pandas_compression)
        elif ser == 'parquet':
            if isinstance(data, pd.Series):
                data = pd.DataFrame({__series__: data})
            data.to_parquet(path, compression=ser_config.pandas_parquet_compression)
        elif ser == 'hdf':
            data.to_hdf(path, os.path.basename(path), mode='w', format='table')
        elif ser == 'json':
            data.to_json(path, compression=ser_config.pandas_compression)
    else:
        raise ImportError(f"Numpy or Pandas are required to serialize data between processes to {path}.")
    return path


@profile(log)
def deserialize_data(path):
    np, pd, sp = _import_data_libraries()
    base, ext = os.path.splitext(path)
    if ext == '.npy':
        if np is None:
            raise ImportError(f"Numpy is required to deserialize {path}.")
        return np.load(path, allow_pickle=ser_config.numpy_pickle)
    elif ext == '.npz':
        _, ext2 = os.path.splitext(base)
        if ext2 == '.spy':
            if sp is None:
                raise ImportError(f"Scipy is required to deserialize {path}.")
            sp_matrix = sp.load_npz(path)
            return unsparsify(sp_matrix, fmt=ser_config.sparse_matrix_deserialized_format)
        else:
            if np is None:
                raise ImportError(f"Numpy is required to deserialize {path}.")
            with np.load(path, allow_pickle=ser_config.numpy_pickle) as loaded:
                return loaded
    else:
        if pd is None:
            raise ImportError(f"Pandas is required to deserialize {path}.")
        ser = ser_config.pandas_serializer
        df = None
        if ser == 'pickle':
            df = pd.read_pickle(path, compression=ser_config.pandas_compression)
        elif ser == 'parquet':
            df = pd.read_parquet(path)
            if len(df.columns) == 1 and df.columns[0] == __series__:
                df = df.squeeze()
        elif ser == 'hdf':
            df = pd.read_hdf(path, os.path.basename(path))
        elif ser == 'json':
            df = pd.read_json(path, compression=ser_config.pandas_compression)
        return unsparsify(df, fmt=ser_config.sparse_dataframe_deserialized_format)

