"""
**datautils** module provide some utility functions for data manipulation.

important
    This is (and should remain) the only non-framework module with dependencies to libraries like pandas or sklearn
    until replacement by simpler/lightweight versions to avoid potential version conflicts with libraries imported by benchmark frameworks.
    Also, this module is intended to be imported by frameworks integration modules,
    therefore, it should have no dependency to any other **amlb** module outside **utils**.
"""
from __future__ import annotations

import logging
import os
from typing import Iterable, Type, Literal, Any, Callable, Tuple, cast, Union
try:
    from typing_extensions import TypeAlias
except ImportError:
    pass  # Only available when dev dependencies are installed, only needed for type check

import arff
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer as Imputer
from sklearn.metrics import accuracy_score, auc, average_precision_score, balanced_accuracy_score, confusion_matrix, fbeta_score, \
    log_loss, mean_absolute_error, mean_squared_error, mean_squared_log_error, precision_recall_curve, \
    r2_score, roc_auc_score  # just aliasing
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OneHotEncoder, OrdinalEncoder

from .utils import profile, path_from_split, repr_def, split_path, touch


log = logging.getLogger(__name__)

A: TypeAlias = Union[np.ndarray, scipy.sparse.csr_matrix]
DF = pd.DataFrame
S = pd.Series

def read_csv(path, nrows=None, header=True, index=False, as_data_frame=True, dtype=None, timestamp_column=None):  # type: ignore  #  Split up to two functions, avoid "aliasing"
    """
    read csv file to DataFrame.

    for now, delegates to pandas, just simplifying signature in the case we want to get rid of pandas dependency
     (numpy should be enough for our needs).
    :param path: the path to a csv file or a file-like object, or readable (with read() method) object.
    :param nrows: the number of rows to read, if not specified, all are read.
    :param header: if the columns header should be read.
    :param as_data_frame: if the result should be returned as a data frame (default) or a numpy array.
    :param dtype: data type for columns.
    :param timestamp_column: name of the column that should be parsed as date.
    :return: a DataFrame
    """
    if timestamp_column is None:
        parse_dates = None
    else:
        if dtype is not None:
            dtype.pop(timestamp_column, None)
        parse_dates = [timestamp_column]
    df = pd.read_csv(path,
                     nrows=nrows,
                     header=0 if header else None,
                     index_col=0 if index else None,
                     dtype=dtype,
                     parse_dates=parse_dates)
    return df if as_data_frame else df.values


def write_csv(  # type: ignore[no-untyped-def]
        data: pd.DataFrame | dict | list | np.ndarray,
        path,
        header: bool = True,
        columns: Iterable[str] | None = None,
        index: bool = False,
        append: bool = False
) -> None:
    if isinstance(data, pd.DataFrame):
        data_frame = data
    else:
        data_frame = to_data_frame(data, column_names=columns)
        header = header and columns is not None
    touch(path)
    data_frame.to_csv(
        path,
        header=header,
        index=index,
        mode=cast(Literal['a','w'], 'a' if append else 'w')
    )


@profile(logger=log)
def reorder_dataset(path: str, target_src: int=0, target_dest: int=-1, save:bool=True) -> str | np.ndarray:
    """ Put the `target_src`th column as the `target_dest`th column"""
    if target_src == target_dest and save:  # no reordering needed, not data to load, returning original path
        return path

    p = split_path(path)
    p.basename += ("_target_" + ("first" if target_dest == 0 else "last" if target_dest == -1 else str(target_dest)))
    reordered_path = path_from_split(p)

    if os.path.isfile(reordered_path):
        if save:  # reordered file already exists, returning it as there's no data to load here
            return reordered_path
        else:  # reordered file already exists, use it to load reordered data
            path = reordered_path

    with open(path) as file:
        df = arff.load(file)

    columns = np.asarray(df['attributes'], dtype=object)
    data = np.asarray(df['data'], dtype=object)

    if target_src == target_dest or path == reordered_path:  # no reordering needed, returning loaded data
        return data

    ori = list(range(len(columns)))
    src = len(columns)+1+target_src if target_src < 0 else target_src
    dest = len(columns)+1+target_dest if target_dest < 0 else target_dest
    if src < dest:
        new = ori[:src]+ori[src+1:dest]+[src]+ori[dest:]
    elif src > dest:
        new = ori[:dest]+[src]+ori[dest:src]+ori[src+1:]
    else:  # no reordering needed, returning loaded data or original path
        return data if not save else path

    reordered_attr = columns[new]
    reordered_data = data[:, new]

    if not save:
        return reordered_data

    with open(reordered_path, 'w') as file:
        arff.dump({
            'description': df['description'],
            'relation': df['relation'],
            'attributes': reordered_attr.tolist(),
            'data': reordered_data.tolist()
        }, file)
    # TODO: provide the possibility to return data even if save is set to false,
    #  as the client code doesn't want to have to load the data again,
    #  and may want to benefit from the caching of reordered data for future runs.
    return reordered_path


def is_data_frame(df: object) -> bool:
    return isinstance(df, pd.DataFrame)


def to_data_frame(obj: object, column_names: Iterable[str]| None=None) -> pd.DataFrame:
    if obj is None:
        return pd.DataFrame()
    columns = None if column_names is None else list(column_names)
    if isinstance(obj, dict):
        orient = cast(Literal['columns', 'index'], 'columns' if columns is None else 'index')
        return pd.DataFrame.from_dict(obj, columns=columns, orient=orient)  # type: ignore[arg-type]
    if isinstance(obj, (list, np.ndarray)):
        return pd.DataFrame.from_records(obj, columns=columns)
    raise ValueError("Object should be a dictionary {col1:values, col2:values, ...} "
                     "or an array of dictionary-like objects [{col1:val, col2:val}, {col1:val, col2:val}, ...].")


class Encoder(TransformerMixin):
    """
    Overly complex "generic" encoder that can handle missing values, auto encoded format (e.g. int for target, float for predictors)...
    Should never have written this, but does the job currently. However, should think about simpler single-purpose approach.
    """

    def __init__(
            self,
            type: Literal['label', 'one-hot','no-op'] ='label',
            target:bool=True,
            encoded_type:Type=int,
            missing_policy:Literal['ignore', 'mask', 'encode']='ignore',
            missing_values: Any | Iterable[Any]| None=None,
            missing_replaced_by: Any='',
            normalize_fn: Callable[[np.ndarray],np.ndarray] | None = None):
        """
        :param type: one of ['label', 'one-hot', 'no-op'].
        :param target: True iff the Encoder is applied to the target feature.
        :param encoded_type: the type of the encoded vec.
        :param missing_policy: one of ['ignore', 'mask', 'encode'].
            ignore: can be safely used only if there's no missing value in the data to be transformed, otherwise it may raise an error during transform().
            mask: replace missing values only internally, and then restore them as np.nan.
            encode: encode all missing values as the encoded value of missing_replaced_by.
        :param missing_values: a value or a list of values considered as missing values.
        :param missing_replaced_by: the value used to replace missing values before encoding.
                                    If using the 'mask' strategy, this is a transient value.
                                    If using the 'encode' strategy, all missing values will be replaced with encode(missing_replaced_by).
        :param normalize_fn: if provided, function applied to all elements during fit and transform (for example, trimming spaces, lowercase...).
        """
        super().__init__()
        assert missing_policy in ['ignore', 'mask', 'encode']
        self.for_target = target
        self.missing_policy = missing_policy
        self.missing_values = set(missing_values).union([None]) if missing_values else {None}
        self.missing_replaced_by = missing_replaced_by
        self.missing_encoded_value = None
        self.normalize_fn = normalize_fn
        self.classes: np.ndarray | None = None
        self.encoded_type = encoded_type
        if type == 'label':
            self.delegate = LabelEncoder() if target else OrdinalEncoder()
        elif type == 'one-hot':
            self.delegate = LabelBinarizer() if target else OneHotEncoder(sparse=False, handle_unknown='ignore')
        elif type == 'no-op':
            self.delegate = None
        else:
            raise ValueError("Encoder `type` should be one of {}.".format(['label', 'one-hot']))

    def __repr__(self) -> str:
        return repr_def(self)

    @property
    def _ignore_missing(self) -> bool:
        return self.for_target or self.missing_policy == 'ignore'

    @property
    def _mask_missing(self) -> bool:
        return not self.for_target and self.missing_policy == 'mask'

    @property
    def _encode_missing(self) -> bool:
        return not self.for_target and self.missing_policy == 'encode'

    def _reshape(self, vec: np.ndarray) -> np.ndarray:
        return vec if self.for_target else vec.reshape(-1, 1)

    def fit(self, vector: Iterable[str] | None) -> 'Encoder':
        """
        :param vector: must be a line vector (array)
        :return:
        """
        if not self.delegate:
            return self

        if vector is None:
            raise ValueError("`vec` can only be `None` if Encoder was initialized with type 'label' or 'one-hot'.")

        vec = np.asarray(vector, dtype=object)
        if self.normalize_fn is not None:
            vec = self.normalize_fn(vec)
        self.classes = np.unique(vec) if self._ignore_missing else np.unique(np.insert(vec, 0, self.missing_replaced_by))

        if self._mask_missing:
            self.missing_encoded_value = self.delegate.fit_transform(self._reshape(cast(np.ndarray, self.classes)))[0]
        else:
            self.delegate.fit(self._reshape(cast(np.ndarray, self.classes)))
        return self

    def transform(self, vec: str | np.ndarray, **params: Any) -> str | np.ndarray:
        """
        :param vec: must be single value (str) or a line vector (array)
        :param params:
        :return:
        """
        if log.isEnabledFor(logging.TRACE):
            log.debug("Transforming %s using %s", vec, self)

        vector = [vec] if isinstance(vec, str) else vec

        def return_value(v: np.ndarray) -> np.ndarray | str:
            return v[0] if isinstance(vec, str) else v

        vector = np.asarray(vector, dtype=object)

        if not self.delegate:
            return return_value(vector.astype(self.encoded_type, copy=False))

        if self._mask_missing or self._encode_missing:
            mask = [v in self.missing_values for v in vector]
            if any(mask):
                # if self._mask_missing:
                #     missing = vector[mask]
                vector[mask] = self.missing_replaced_by
                if self.normalize_fn:
                    vector = self.normalize_fn(vector)

                res = self.delegate.transform(self._reshape(vector), **params).astype(self.encoded_type, copy=False)
                if self._mask_missing:
                    res[mask] = np.NaN if self.encoded_type == float else None
                return return_value(res)

        if self.normalize_fn:
            vector = self.normalize_fn(vector)
        return return_value(self.delegate.transform(self._reshape(vector), **params).astype(self.encoded_type, copy=False))

    def inverse_transform(self, vec: str | np.ndarray, **params: Any) -> str | np.ndarray:
        """
        :param vec: must a single value or line vector (array)
        :param params:
        :return:
        """
        if not self.delegate:
            return vec

        # TODO: handle mask
        vec = np.asarray(vec).astype(self.encoded_type, copy=False)
        return self.delegate.inverse_transform(vec, **params)


def impute_array(
        X_fit: A,
        *X_s: Iterable[A],
        missing_values: Any =np.NaN,
        strategy: Literal['mean', 'mode', 'median', 'most_frequent'] | Tuple[Literal['constant'],  Any]="mean",
        keep_empty_features: bool = False) -> list[A] | A:
    """
    :param X_fit: {array-like, sparse matrix} used to fit the imputer. This array is also imputed.
    :param X_s: the additional (optional) arrays that are imputed using the same imputer.
    :param missing_values: the value that will be substituted during the imputation.
    :param strategy: 'mean' (default) -> missing values are imputed with the mean value of the corresponding vector.
                     'median' -> missing values are imputed with the median value of the corresponding vector.
                     'mode' -> missing values are imputed with the mode of the corresponding vector.
                     'most_frequent' -> alias for 'mode'
                     ('constant', value) -> missing values are imputed with the constant value provided as the second term of the tuple.
                     None -> no-op (for internal use).
    :param keep_empty_features: bool (default False), if False remove all columns which only have nan values.
    :return: a list of imputed arrays, returned in the same order as they were provided.
    """
    if strategy is None:
        return [X_fit, *X_s]
    strategy_name, fill_value = strategy if isinstance(strategy, tuple) and strategy[0] == 'constant' else (strategy, None)
    strategy_name = dict(mode='most_frequent').get(strategy_name, strategy_name)  # type: ignore

    imputer = Imputer(missing_values=missing_values, strategy=strategy_name, fill_value=fill_value, keep_empty_features=keep_empty_features)
    imputed = _restore_dtypes(imputer.fit_transform(X_fit), X_fit)
    if len(X_s) > 0:
        result = [imputed]
        for X in X_s:
            result.append(_restore_dtypes(imputer.transform(X), X))  #  type: ignore
        return result
    return imputed


def impute_dataframe(X_fit: pd.DataFrame, *X_s: Iterable[pd.DataFrame], missing_values: Any=np.NaN,
                     strategy: Literal['mean','median','mode'] | Tuple[Literal['constant'], Any] ='mean'
                    ) -> pd.DataFrame | list[pd.DataFrame]:
    """
    :param X_fit: used to fit the imputer. This dataframe is also imputed.
    :param X_s: the additional (optional) dataframe that are imputed using the same imputer.
    :param missing_values: the value that will be substituted during the imputation.
    :param strategy: 'mean' (default) -> missing values are imputed with the mean value of the corresponding vector.
                     'median' -> missing values are imputed with the median value of the corresponding vector.
                     'mode' -> missing values are imputed with the mode of the corresponding vector.
                     ('constant', value) -> missing values are imputed with the constant value provided as the second term of the tuple.
                     None -> no-op (for internal use).
                     { type: strategy } -> (not ready yet!) each column/feature type will be applied a different strategy as soon as it is listed in the dictionary.
                                           type must be one of (int, float, number, bool, category, string, object, datetime)
    :return: a list of imputed dataframes, returned in the same order as they were provided.
    """
    if strategy is None:
        return [X_fit, *X_s]
    if isinstance(strategy, dict):
        for dt, s in strategy.items():
            X_fit.select_dtypes(include=dt)
    else:
        imputed = _impute_pd(X_fit, *X_s, missing_values=missing_values, strategy=strategy)
    return imputed if X_s else imputed[0]


def _impute_pd(
        X_fit: pd.DataFrame,
        *X_s: Iterable[pd.DataFrame],
        missing_values: Any = np.NaN,
        strategy: Literal['mean','median','mode'] | Tuple[Literal['constant'], Any] | None =None,
        is_int: bool = False
) -> list[pd.DataFrame]:
    if strategy == 'mean':
        fill = X_fit.mean()
    elif strategy == 'median':
        fill = X_fit.median()
    elif strategy == 'mode':
        fill = X_fit.mode().iloc[0, :]  # type: ignore[call-overload]
    elif isinstance(strategy, tuple) and strategy[0] == 'constant':
        fill = strategy[1]
    else:
        return [X_fit, *X_s]  # type: ignore[list-item]  # doesn't seem to understand unpacking

    if is_int and isinstance(fill, pd.Series):
        fill = fill.round()
    return [df.replace(missing_values, fill) for df in [X_fit, *X_s]]


def _rows_with_nas(X: np.ndarray | pd.DataFrame) -> pd.DataFrame:
    df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    return df[df.isna().any(axis=1)]


def _restore_dtypes(X_np: np.ndarray, X_ori: pd.DataFrame | pd.Series | np.ndarray) -> pd.DataFrame | pd.Series |  np.ndarray:
    if isinstance(X_ori, pd.DataFrame):
        df = pd.DataFrame(X_np, columns=X_ori.columns, index=X_ori.index).convert_dtypes()
        df.astype(X_ori.dtypes.to_dict(), copy=False, errors='raise')
        return df
    elif isinstance(X_ori, pd.Series):
        return pd.Series(X_np, name=X_ori.name, index=X_ori.index, dtype=X_ori.dtype)
    elif isinstance(X_ori, np.ndarray):
        return X_np.astype(X_ori.dtype, copy=False)
    else:
        return X_np
