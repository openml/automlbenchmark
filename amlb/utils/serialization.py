import logging
import math
import os
import pickle
import re
from typing import Optional

from .core import Namespace as ns, json_dump, json_load
from .process import profile

log = logging.getLogger(__name__)

__no_export = set(dir())  # all variables defined above this are not exported


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
        import scipy.sparse as sp  # type: ignore # https://github.com/scipy/scipy/issues/17158
    except ImportError:
        sp = None
    return np, pd, sp


ser_config = ns(
    # the serializer to use when there's no specific serializer available.
    # mainly intended to serialize simple data structures like lists.
    # allowed=['pickle', 'json']
    fallback_serializer='json',
    # if numpy can use pickle to serialize ndarrays,
    numpy_allow_pickle=True,
    # format used to serialize pandas dataframes/series between processes.
    # allowed=['pickle', 'parquet', 'hdf', 'json']
    pandas_serializer='parquet',
    # the compression format used when serializing pandas dataframes/series.
    # allowed=[None, 'infer', 'bz2', 'gzip']
    # 'infer' (= None) is the fastest but no compression,
    # 'gzip' fast write and read with good compression.
    # 'bz2' looks like the best compression/time ratio (faster write, sometimes slightly slower read)
    pandas_compression='infer',
    # the compression format used when serializing pandas dataframes/series to parquet.
    # allowed=[None, 'snappy', 'gzip', 'brotli']
    pandas_parquet_compression=None,
    # if sparse matrices should be compressed during serialization.
    sparse_matrix_compression=True,
    # if sparse matrices should be deserialized to some specific format:
    # allowed=[None, 'array', 'dense']
    #  None (no change), 'array' (numpy), 'dense' (dense matrix).
    sparse_matrix_deserialized_format=None,
    # if sparse dataframes should be deserialized to some specific format:
    # allowed=[None, 'array', 'dense']
    #  None (no change), 'array' (numpy), 'dense' (dense dataframe/series).
    sparse_dataframe_deserialized_format=None,
)

__series__ = '_series_'


class SerializationError(Exception):
    pass


def is_serializable_data(data):
    np, pd, sp = _import_data_libraries()
    return isinstance(data, (np.ndarray, sp.spmatrix, pd.DataFrame, pd.Series))


def is_sparse(data):
    np, pd, sp = _import_data_libraries()
    return ((sp and isinstance(data, sp.spmatrix))   # sparse matrix
            or (pd and isinstance(data, pd.Series) and pd.api.types.is_sparse(data.dtype))  # sparse Series
            or (pd and isinstance(data, pd.DataFrame)  # if one column is sparse, the dataframe is considered as sparse
                and any(pd.api.types.is_sparse(dt) for dt in data.dtypes)))


def unsparsify(*data, fmt='dense'):
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
        return (data.to_numpy(copy=False) if fmt == 'array'
                else _pd_to_dense(pd, data) if fmt == 'dense' and is_sparse(data)
                else data)
    else:
        return data


def _pd_to_dense(pd, df):
    if hasattr(df, 'sparse'):
        return df.sparse.to_dense()
    data = {k: (v.sparse.to_dense() if hasattr(v, 'sparse') else v) for k, v in df.items()}
    return pd.DataFrame(data, index=df.index, columns=df.columns)


def _pd_dtypes_to_str(pd, df):
    return {k: str(v) for k, v in df.dtypes.items()}


def _pd_dtypes_from_str(pd, dt):
    def dt_from_str(s):
        m_sparse = re.match(r"Sparse\[(.*)]", s)
        if m_sparse:
            sub_type, fill_value = [t.strip() for t in m_sparse.group(1).split(",", 1)]
            try:
                fill_value = eval(fill_value, {'nan': math.nan, '<NA>': pd.NA})
            except ValueError:
                pass
            dt = pd.api.types.pandas_dtype(f"Sparse[{sub_type}]")
            return pd.SparseDtype(dt, fill_value=fill_value)
        else:
            return pd.api.types.pandas_dtype(s)

    return {k: dt_from_str(v) for k, v in dt.items()}


@profile(log)
def serialize_data(data, path, config: Optional[ns] = None):
    config = (config | ser_config) if config else ser_config
    root, ext = os.path.splitext(path)
    np, pd, sp = _import_data_libraries()
    if np and isinstance(data, np.ndarray):
        path = f"{root}.npy"
        np.save(path, data, allow_pickle=config.numpy_allow_pickle)
    elif sp and isinstance(data, sp.spmatrix):
        # use custom extension to recognize sparsed matrices from file name.
        # .npz is automatically appended if missing, and can also potentially be used for numpy arrays.
        path = f"{root}.spy.npz"
        sp.save_npz(path, data, compressed=config.sparse_matrix_compression)
    elif pd and isinstance(data, (pd.DataFrame, pd.Series)):
        path = f"{root}.pd"
        if isinstance(data, pd.DataFrame):
            # pandas has this habit of inferring value types when data are loaded from file,
            # for example, 'true' and 'false' are converted automatically to booleans, even for column names…
            data.rename(str, axis='columns', inplace=True)
        ser = config.pandas_serializer
        if ser == 'pickle':
            data.to_pickle(path, compression=config.pandas_compression)
        elif ser == 'parquet':
            if isinstance(data, pd.Series):
                data = pd.DataFrame({__series__: data})
            # parquet serialization doesn't support sparse dataframes
            if is_sparse(data):
                path = f"{root}.sparse.pd"
                dtypes = _pd_dtypes_to_str(pd, data)
                json_dump(dtypes, f"{path}.dtypes", style='compact')
                data = unsparsify(data)
            data.to_parquet(path, compression=config.pandas_parquet_compression)
        elif ser == 'hdf':
            data.to_hdf(path, os.path.basename(path), mode='w', format='table')
        elif ser == 'json':
            data.to_json(path, compression=config.pandas_compression)
    else:  # fallback serializer
        if config.fallback_serializer == 'json':
            path = f"{root}.json"
            json_dump(data, path, style='compact')
        else:
            path = f"{root}.pkl"
            with open(path, 'wb') as f:
                pickle.dump(data, f)
    return path


@profile(log)
def deserialize_data(path, config: Optional[ns] = None):
    config = (config | ser_config) if config else ser_config
    np, pd, sp = _import_data_libraries()
    base, ext = os.path.splitext(path)
    if ext == '.npy':
        if np is None:
            raise SerializationError(f"Numpy is required to deserialize {path}.")
        return np.load(path, allow_pickle=config.numpy_allow_pickle)
    elif ext == '.npz':
        _, ext2 = os.path.splitext(base)
        if ext2 == '.spy':
            if sp is None:
                raise SerializationError(f"Scipy is required to deserialize {path}.")
            sp_matrix = sp.load_npz(path)
            return unsparsify(sp_matrix, fmt=config.sparse_matrix_deserialized_format)
        else:
            if np is None:
                raise SerializationError(f"Numpy is required to deserialize {path}.")
            with np.load(path, allow_pickle=config.numpy_pickle) as loaded:
                return loaded
    elif ext == '.pd':
        if pd is None:
            raise SerializationError(f"Pandas is required to deserialize {path}.")
        ser = config.pandas_serializer
        df = None
        if ser == 'pickle':
            df = pd.read_pickle(path, compression=config.pandas_compression)
        elif ser == 'parquet':
            df = pd.read_parquet(path)
            if len(df.columns) == 1 and df.columns[0] == __series__:
                df = df.squeeze()
            _, ext2 = os.path.splitext(base)
            if config.sparse_dataframe_deserialized_format is None and ext2 == '.sparse':
                # trying to restore dataframe as sparse if it was as such before serialization
                # and if the dataframe format should remain unchanged
                j_dtypes = json_load(f"{path}.dtypes")
                dtypes = _pd_dtypes_from_str(pd, j_dtypes)
                df = df.astype(dtypes, copy=False)
        elif ser == 'hdf':
            df = pd.read_hdf(path, os.path.basename(path))
        elif ser == 'json':
            df = pd.read_json(path, compression=config.pandas_compression)
        return unsparsify(df, fmt=config.sparse_dataframe_deserialized_format)
    elif ext == '.json':
        return json_load(path)
    elif ext == '.pkl':
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        raise SerializationError(f"Can not deserialize file `{path}` in unknown format.")


__all__ = [s for s in dir() if not s.startswith('_') and s not in __no_export]
