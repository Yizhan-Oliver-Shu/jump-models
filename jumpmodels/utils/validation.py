"""
utils to validate function inputs and parameters.
"""

import numpy as np
import pandas as pd
import numbers
from typing import Union

PD_TYPE = Union[pd.Series, pd.DataFrame]
NUMERICAL_OBJ_TYPE = Union[np.ndarray, PD_TYPE]
SER_ARR_TYPE = Union[np.ndarray, pd.Series]
DF_ARR_TYPE = Union[np.ndarray, pd.DataFrame]

pd.set_option('display.width', 300)

# reviewed
def is_no_nan(obj):
    """
    return whether an object doesn't contain any nan/None value.
    """
    return not pd.isna(np.asarray(obj)).any()

# reviewed
def valid_no_nan(obj):
    """
    assert that an object doesn't contain any nan/None value.
    """
    assert is_no_nan(obj), f"contains NaNs."

# reviewed
def check_2d_array(X: NUMERICAL_OBJ_TYPE, dtype=None, assert_na=True) -> np.ndarray:
    """
    check an array-like object into a 2-d array. 1-d objects will be appended a new axis. only accepts 1-d and 2-d inputs.
    returns a copy for data safety.

    Parameters:
    ---------------------------
    X: array/series/dataframe
        
    dtype:
        dtype for the return arr
    assert_na: bool
        if `True` will assert that `X` doesn't contain any NaNs.
    """
    X = np.array(X, dtype=dtype)
    if X.ndim == 1: X = X[:, np.newaxis]    # append new axis
    assert X.ndim == 2
    if assert_na: valid_no_nan(X)
    return X

# reviewed
def check_1d_array(X: NUMERICAL_OBJ_TYPE, dtype=None, assert_na=True) -> np.ndarray:
    """
    check an array/series/dataframe into a 1d array.
    returns a copy for data safety.

    Parameters:
    ---------------------------
    X: array/series/dataframe
        
    dtype:
        dtype for the return arr
    assert_na: bool
        if `True` will assert that `X` doesn't contain any NaNs.
    """
    X = np.array(X, dtype=dtype).squeeze()
    assert X.ndim == 1
    if assert_na: valid_no_nan(X)
    return X

# reviewed
def is_ser_df(obj):
    """
    return whether an object is a pd Series/DataFrame.
    """
    return isinstance(obj, PD_TYPE) 

# reviewed
def is_numbers(x):
    """
    whether x is a strict scalar number.
    """
    return isinstance(x, numbers.Number)

# reviewed
def check_string(string):
    """
    if None, return "", else return itself as a tring.
    """
    if string is None: return ""
    return str(string)

# reviewed
def is_rows_all_True_or_all_False(x: np.ndarray):
    """
    return whether all of the rows of a 2d boolean arr are either all True or all False.
    """
    assert x.ndim == 2
    return np.logical_or(x.all(axis=1), ~(x.any(axis=1))).all()

# reviewed
def check_datetime_date(date):
    """
    convert any date-like object into a `datetime.date()` object. if None, return None.
    """
    if date is None: return None
    return pd.Timestamp(date).date()

###############################
## function output
###############################

# reviewed
def raise_arr_to_pd_obj(arr: np.ndarray, pd_obj: NUMERICAL_OBJ_TYPE, index_key="index", columns_key=None, return_as_ser=True) -> NUMERICAL_OBJ_TYPE:
    """
    raise an arr into pd ser/df, from the index/columns of `pd_obj`. if `pd_obj` is not pd objects, return `arr` itself.

    Parameters:
    ---------------------------
    arr: array
        
    pd_obj:
        if not a pd obj, `arr` will be returned.
    index_key:
        the attribute `index_key` of `pd_obj` will be used as the index for the return value
    columns_key:
        for `columns` attr if needed
    return_as_ser:
        if `True` the return value is a ser, else a df.
    """
    if not is_ser_df(pd_obj): return arr
    def getattr_(obj, key):
        if key is None: return None
        assert hasattr(obj, key)
        return getattr(obj, key)
    index = getattr_(pd_obj, index_key)
    columns = getattr_(pd_obj, columns_key)
    if return_as_ser: return pd.Series(arr, index=index)
    return pd.DataFrame(arr, index=index, columns=columns)

###############################
## dates
###############################

# reviewed
def filter_date_range(obj: PD_TYPE, start_date=None, end_date=None) -> PD_TYPE:
    """
    filter a pd ser/df with `datetime.date()` index, by a date range.
    """
    assert is_ser_df(obj)
    start_date, end_date =  check_datetime_date(start_date), check_datetime_date(end_date)
    if start_date is not None: obj = obj.loc[start_date:]
    if end_date is not None: obj = obj.loc[:end_date]
    return obj.copy()

# reviewed
def align_index(x: PD_TYPE, y: PD_TYPE) -> PD_TYPE:
    """
    return a subset of x so that the index aligns with that of y.
    """
    return x.loc[y.index].copy()
