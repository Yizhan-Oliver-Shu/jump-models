"""
Helpers for working with the index of pandas objects, typically of type `datetime.date`.

This module provides functions to filter and align the index of pandas Series 
and DataFrames. The functionality ensures proper handling of date-based indices and 
alignment of pandas objects.

Depends on
----------
utils.validation : Module
"""

from .validation import *

# reviewed
def filter_date_range(obj: PD_TYPE, start_date: DATE_TYPE = None, end_date: DATE_TYPE = None) -> PD_TYPE:
    """
    Filter a pandas Series or DataFrame with a `datetime.date` index by a specified date range.
    Returns a copy of the filtered object for data safety.

    Parameters
    ----------
    obj : Series or DataFrame
        The pandas object to filter, which must have an index of dtype `datetime.date`.

    start_date : str, datetime.date, or None, optional
        The start date of the range. If `None`, no start date filter is applied.

    end_date : str, datetime.date, or None, optional
        The end date of the range. If `None`, no end date filter is applied.

    Returns
    -------
    Series or DataFrame
        A copy of the filtered pandas object.
    """
    assert is_ser_df(obj)
    start_date, end_date =  check_datetime_date(start_date), check_datetime_date(end_date)
    if start_date is not None: obj = obj.loc[start_date:]
    if end_date is not None: obj = obj.loc[:end_date]
    return obj.copy()

# reviewed
def align_index(x: PD_TYPE, y: PD_TYPE) -> PD_TYPE:
    """
    Return a subset of `x` so that its index aligns with the index of `y`. 
    Returns a copy of the subset for data safety.

    Parameters
    ----------
    x : Series or DataFrame
        The pandas object whose index is to be aligned with `y`.

    y : Series or DataFrame
        The pandas object whose index is used for alignment.

    Returns
    -------
    Series or DataFrame
        A copy of `x` with its index aligned to `y`.
    """
    return x.loc[y.index].copy()    # throw error if the index is not contained

# reviewed
def align_x_with_y(x: NUMERICAL_OBJ_TYPE, y: NUMERICAL_OBJ_TYPE) -> NUMERICAL_OBJ_TYPE:
    """
    Align `x` with `y`. If both `x` and `y` are pandas objects, align their indices using 
    `align_index`. If they are not both pandas objects, assert that their lengths match.
    Returns a copy for data safety.

    Parameters
    ----------
    x : ndarray, Series, or DataFrame
        The first numerical object to align.

    y : ndarray, Series, or DataFrame
        The second numerical object to align.

    Returns
    -------
    ndarray, Series, or DataFrame
        A copy of `x`, aligned with `y`.
    """
    if is_ser_df(x) and is_ser_df(y): return align_index(x, y)
    # not all pd objects, assert that lens match
    assert is_same_len(x, y), "the two input arrays should be of the same length"
    return x.copy()
