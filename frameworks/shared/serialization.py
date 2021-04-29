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
    return np, pd


ser_config = ns(
    # 'pickle', 'parquet', 'hdf', 'json'
    pandas_serializer=os.environ.get('AMLB_SER_PD_MODE') or 'pickle',
    # 'infer', 'bz2', 'gzip', None ?
    # 'infer' (here=None) is the fastest but no compression,
    # 'gzip' fast write and read with good compression.
    # 'bz2' looks like the best compression/time ratio (faster write, sometimes slightly slower read)
    pandas_compression=os.environ.get('AMLB_SER_PD_COMPR') or 'infer',
    # 'snappy', 'gzip', 'brotli', None ?
    pandas_parquet_compression=os.environ.get('AMLB_SER_PD_PQT_COMPR') or None,
)


__series__ = '_series_'


@profile(log)
def serialize_data(data, path):
    np, pd = _import_data_libraries()
    if pd and isinstance(data, (pd.DataFrame, pd.Series)):
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
    elif np:
        root, ext = os.path.splitext(path)
        path = f"{root}.npy"
        np.save(path, data, allow_pickle=True)
    else:
        raise ImportError("Numpy or Pandas are required to serialize data between processes.")
    return path


@profile(log)
def deserialize_data(path):
    np, pd = _import_data_libraries()
    _, ext = os.path.splitext(path)
    if ext == ".npy":
        if np is None:
            raise ImportError("Numpy is required to deserialize data.")
        return np.load(path, allow_pickle=True)
    else:
        if pd is None:
            raise ImportError("Pandas is required to deserialize data.")
        ser = ser_config.pandas_serializer
        if ser == 'pickle':
            return pd.read_pickle(path, compression=ser_config.pandas_compression)
        elif ser == 'parquet':
            df = pd.read_parquet(path)
            if len(df.columns) == 1 and df.columns[0] == __series__:
                return df.squeeze()
            return df
        elif ser == 'hdf':
            return pd.read_hdf(path, os.path.basename(path))
        elif ser == 'json':
            return pd.read_json(path, compression=ser_config.pandas_compression)

