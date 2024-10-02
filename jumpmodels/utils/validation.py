"""
Module of functions to validate input/output and parameters in functions or estimators.

This module provides general validation functions and does not depend on any custom modules.
"""

import numpy as np
import pandas as pd
import numbers
from typing import Union, Optional, Dict
import datetime

# custom data types
PD_TYPE = Union[pd.Series, pd.DataFrame]
NUMERICAL_OBJ_TYPE = Union[np.ndarray, PD_TYPE]
SER_ARR_TYPE = Union[np.ndarray, pd.Series]
DF_ARR_TYPE = Union[np.ndarray, pd.DataFrame]
DATE_TYPE = Optional[Union[str, datetime.date]]

pd.set_option('display.width', 300)

###############################
## convert input types 
###############################

# reviewed
def is_no_nan(obj: NUMERICAL_OBJ_TYPE) -> bool:
    """
    Check whether an object does not contain any NaN or None values.

    Parameters
    ----------
    obj : Array/Series/DataFrame
        The input numerical object to check. It can be a numpy array, pandas Series, 
        or pandas DataFrame.

    Returns
    -------
    bool
        `True` if the object does not contain any NaN or None values, `False` otherwise.
    """
    return not pd.isna(np.asarray(obj)).any()

# reviewed
def valid_no_nan(obj: NUMERICAL_OBJ_TYPE):
    """
    Assert that an object does not contain any NaN or None values.

    Parameters
    ----------
    obj : Array/Series/DataFrame
        The input numerical object to check. It can be a numpy array, pandas Series, 
        or pandas DataFrame.

    Raises
    ------
    AssertionError
        If the object contains NaN or None values.
    """
    assert is_no_nan(obj), f"input numerical object contains NaNs."
    return 

# reviewed
def check_2d_array(X: NUMERICAL_OBJ_TYPE, single_col=False, dtype=None, assert_na=True) -> np.ndarray:
    """
    Convert an array-like object into a 2D array. If the input is 1D, a new axis will be appended.
    Only accepts 1D and 2D inputs. If `single_col` is True, the function will assert that 
    `X.shape[1] == 1`. The function returns a copy for data safety.

    Parameters
    ----------
    X : Array/Series/DataFrame
        Array-like object (numpy array, pandas Series, or pandas DataFrame). Raises an exception if 
        the dimensionality is not 1 or 2.
    
    single_col : bool, optional (default=False)
        If True, assert that `X.shape[1] == 1`, ensuring that the input contains only one column.

    dtype : data-type, optional
        Desired numpy data type for the returned array.

    assert_na : bool, optional (default=True)
        Whether to assert that the input `X` does not contain any NA values.

    Returns
    -------
    np.ndarray
        A 2D numpy array.
    """
    X = np.array(X, dtype=dtype)
    if X.ndim == 1: X = X[:, np.newaxis]    # append new axis
    assert X.ndim == 2
    if single_col: assert X.shape[1] == 1
    if assert_na: valid_no_nan(X)
    return X

# reviewed
def check_1d_array(X: NUMERICAL_OBJ_TYPE, dtype=None, assert_na=True) -> np.ndarray:
    """
    Convert an array-like object into a 1D array. The function returns a copy for data safety.

    Parameters
    ----------
    X : Array/Series/DataFrame
        Array-like object (numpy array, pandas Series, or pandas DataFrame). Raises an exception if 
        the dimensionality after calling `.squeeze()` is not 1.

    dtype : data-type, optional
        Desired numpy data type for the returned array.

    assert_na : bool, optional (default=True)
        Whether to assert that the input `X` does not contain any NA values.

    Returns
    -------
    np.ndarray
        A 1D numpy array.
    """
    X = np.array(X, dtype=dtype).squeeze()
    assert X.ndim == 1
    if assert_na: valid_no_nan(X)
    return X

# reviewed
def check_datetime_date(date: DATE_TYPE) -> Optional[datetime.date]:
    """
    Convert a date-like object into a `datetime.date` object. If the input is `None`, 
    return `None`.

    Parameters
    ----------
    date : str, datetime.date, or None
        The input date-like object to be converted. Can be a string, a datetime object, 
        or `None`.

    Returns
    -------
    datetime.date or None
        A `datetime.date` object if the input is a valid date-like object, otherwise `None`.
    """
    if date is None: return None
    return pd.Timestamp(date).date()

###############################
## binary checks
###############################

# reviewed
def is_ser(obj) -> bool:
    """
    Check whether the input object is a Series.
    """
    return isinstance(obj, pd.Series)

# reviewed
def is_df(obj) -> bool:
    """
    Check whether the input object is a DataFrame.
    """
    return isinstance(obj, pd.DataFrame)

# reviewed
def is_ser_df(obj) -> bool:
    """
    Check whether the input object is a Series/DataFrame.
    """
    return isinstance(obj, PD_TYPE) 

# reviewed
def is_numbers(x) -> bool:
    """
    Check whether the input is a scalar number.
    """
    return isinstance(x, numbers.Number)

# reviewed
def is_same_len(*args) -> bool:
    """
    Check whether all input arguments have the same length.

    Parameters
    ----------
    *args : iterable
        Variable number of input iterables (e.g., lists, arrays, or other iterable objects).

    Returns
    -------
    bool
        `True` if all input arguments have the same length, `False` otherwise.
    """
    return len(set(len(x) for x in args)) == 1

# reviewed
def is_same_index(*args) -> bool:
    """
    Check whether the index of all input pandas Series or DataFrames are exactly the same.
    This function is typically used to verify if the date indices of different Series/DataFrames 
    align with each other.

    Parameters
    ----------
    *args : Series or DataFrame
        Variable number of pandas Series or DataFrame objects whose indices are to be compared.

    Returns
    -------
    bool
        `True` if all input Series/DataFrames have the same index, `False` otherwise.
    """
    assert is_same_len(*args)
    index_this = None
    for item in args:
        # assert is_ser_df(item)
        if index_this is None: # the first item
            index_this = item.index
            continue 
        index_that = item.index
        if not (index_this==index_that).all():
            return False
    return True

###############################
## output cast in pd types 
###############################

# reviewed
def getattr_(obj: object, key: Optional[str]): 
    """
    Retrieve the attribute `key` from the object `obj`. If `key` is `None`, or the object 
    does not have the attribute `key`, return `None`.

    Parameters
    ----------
    obj : object
        The object from which to retrieve the attribute.

    key : str, optional
        The name of the attribute to retrieve. If `None`, the function returns `None`.

    Returns
    -------
    any or None
        The value of the attribute if it exists, otherwise `None`.
    """
    if key is not None and hasattr(obj, key):
        return getattr(obj, key) 
    else:
        return None

# reviewed
def raise_arr_to_pd_obj(arr: np.ndarray, pd_obj: NUMERICAL_OBJ_TYPE, index_key="index", columns_key="columns", return_as_ser=True) -> NUMERICAL_OBJ_TYPE:
    """
    Convert a numpy array into a pandas Series or DataFrame, using the index and columns 
    attributes of `pd_obj` for labeling. If `pd_obj` is not a pandas object, the function 
    returns the array unchanged.

    Parameters
    ----------
    arr : np.ndarray
        The array to be converted into a pandas Series or DataFrame.

    pd_obj : Series, DataFrame, or array-like
        The pandas object from which to extract the index and columns for the new pandas object.

    index_key : str, optional (default="index")
        The attribute name for retrieving the index of the output from `pd_obj`.

    columns_key : str, optional (default="columns")
        The attribute name for retrieving the columns of the output from `pd_obj`.
        Only useful if the parameter `return_as_ser` is set to `False`.

    return_as_ser : bool, optional (default=True)
        If `True`, the function returns a pandas Series using only the index. 
        If `False`, it returns a pandas DataFrame using both the index and columns.

    Returns
    -------
    Series, DataFrame, or np.ndarray
        A pandas Series or DataFrame with index and columns matching those of `pd_obj`, 
        or the original numpy array if `pd_obj` is not a pandas object.
    """
    if not is_ser_df(pd_obj): return arr
    index = getattr_(pd_obj, index_key)
    columns = getattr_(pd_obj, columns_key)
    if return_as_ser: return pd.Series(arr, index=index)
    return pd.DataFrame(arr, index=index, columns=columns)

###############################
## file i/o
###############################

import os

# reviewed
def check_dir_exist(filepath):
    """
    Check whether the directory of the specified file path exists. If it does not exist, 
    create the directory. Handles potential race conditions where multiple processes may 
    attempt to create the directory simultaneously.

    Parameters
    ----------
    filepath : str
        The file path for which the existence of the parent directory is checked.
    """
    dirname = os.path.dirname(filepath)
    if dirname != "":
        if not os.path.exists(dirname):
            try:
                os.makedirs(dirname, exist_ok=True)
                print(f"Created folder: {dirname}")
            except FileExistsError:
                # The directory was created by another process between the check and creation
                pass
    return
