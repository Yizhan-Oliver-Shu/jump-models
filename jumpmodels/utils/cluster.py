"""
Helpers for numerical calculations in clustering analysis.

This module provides functions to handle clustering-related tasks such as label validation, 
probability conversion, and transition matrix computation.

Depends on
----------
utils.validation : Module
"""

from .validation import *

# reviewed
def is_valid_labels(labels_: SER_ARR_TYPE, n_c: int = 2) -> bool:
    """
    Check whether a label array/series is a valid label sequence. The values of `labels_` must 
    lie in the set {0, 1, ..., n_c-1}.

    Parameters
    ----------
    labels_ : ndarray or Series
        The array or series of labels to check.

    n_c : int, optional (default=2)
        The number of clusters. Labels must lie in {0, 1, ..., n_c-1}.

    Returns
    -------
    bool
        True if the labels are valid, False otherwise.
    """
    labels_arr = check_1d_array(labels_)   # check whether it is intrinsically 1-d
    return set(labels_arr).issubset(set(range(n_c)))

# reviewed
def is_valid_proba(proba_: DF_ARR_TYPE) -> bool:
    """
    Check whether a probability array/series is valid, meaning all values are non-negative 
    and all rows sum to 1.

    Parameters
    ----------
    proba_ : ndarray or DataFrame
        The probability matrix to check.

    Returns
    -------
    bool
        True if the probability matrix is valid, False otherwise.
    """
    proba_arr = check_2d_array(proba_)
    return (proba_arr>=0).all() and np.isclose(proba_arr.sum(axis=1), 1.).all()

# reviewed
def raise_labels_into_proba(labels_: np.ndarray, n_c: int) -> np.ndarray:
    """
    Convert a discrete label array into a probability matrix. The resulting matrix corresponds 
    to hard clustering, with 0./1. values.

    Parameters
    ----------
    labels_ : ndarray of shape (n_s,)
        The array of integer labels.

    n_c : int
        The number of clusters.

    Returns
    -------
    proba_ : ndarray of shape (n_s, n_c)
        The probability assignment array.
    """
    # labels_ must be ints, and smaller than n_c
    # don't verify inputs, for performance consideration
    n_s = len(labels_)
    proba_ = np.zeros((n_s, n_c)) 
    proba_[range(n_s), labels_] = 1.
    # assert is_valid_proba(proba_)
    return proba_

# reviewed
def reduce_proba_to_labels(proba_: DF_ARR_TYPE) -> SER_ARR_TYPE:
    """
    Convert a probability matrix into a label series by taking the argmax of each row.

    Parameters
    ----------
    proba_ : ndarray or DataFrame
        The probability matrix to convert.

    Returns
    -------
    labels_ : ndarray or Series
        The label series obtained by taking the argmax of each row.
    """
    if is_df(proba_): return proba_.idxmax(axis=1)
    # arr
    return proba_.argmax(axis=1)

# reviewed
def is_map_from_left_to_right(labels_left: Optional[SER_ARR_TYPE], labels_right: Optional[SER_ARR_TYPE]) -> bool:
    """
    Check whether the map from `labels_left` to `labels_right` is valid, meaning elements with the same label 
    in `labels_left` must have the same label in `labels_right`. If either label array is `None`, return `False`.

    Parameters
    ----------
    labels_left : ndarray or Series, optional
        The left-side label array.

    labels_right : ndarray or Series, optional
        The right-side label array.

    Returns
    -------
    bool
        True if the mapping is valid, False otherwise.
    """
    if labels_left is None or labels_right is None:
        return False
    assert len(labels_left) == len(labels_right)
    for label in np.unique(labels_left):
        if len(np.unique(labels_right[labels_left==label])) != 1:
            return False
    return True

# reviewed
def is_same_clustering(labels1: Optional[SER_ARR_TYPE], labels2: Optional[SER_ARR_TYPE]) -> bool:
    """
    Check whether two clustering results are the same, under permutation. If either input is `None`, return `False`.

    Parameters
    ----------
    labels1 : ndarray or Series, optional
        The first label array.

    labels2 : ndarray or Series, optional
        The second label array.

    Returns
    -------
    bool
        True if the two clustering results are the same, False otherwise.
    """
    return is_map_from_left_to_right(labels1, labels2) and is_map_from_left_to_right(labels2, labels1)

# reviewed
def empirical_trans_mx(labels_: SER_ARR_TYPE, n_components=2, return_counts=False) -> np.ndarray:
    """
    Compute the empirical transition count or probability matrix from a label array/series. 
    Probability values will be `nan` if no transition from a state is observed.

    Parameters
    ----------
    labels_ : ndarray or Series
        The label array/series with values in {0, 1, ..., n_components - 1}, of both float/int dtype.

    n_components : int, optional (default=2)
        The number of unique labels.

    return_counts : bool, optional (default=False)
        If True, return the transition counts instead of probabilities.

    Returns
    -------
    ndarray
        The transition count or probability matrix.
    """
    assert is_valid_labels(labels_, n_c=n_components)
    labels_ = check_1d_array(labels_, dtype=int)    # labels must be int type, as it will be used as arr index.
    # count transitions
    count_mx = np.zeros((n_components, n_components), dtype=int)
    for i in range(n_components):
        # the next states after label==i
        labels_next = labels_[1:][labels_[:-1]==i]  # shift label by 1
        # count next states
        states, counts = np.unique(labels_next, return_counts=True)     # states must be ints.
        count_mx[i, states] = counts
    if return_counts: return count_mx
    # return probability
    return (1.*count_mx) / count_mx.sum(axis=1, keepdims=True)

# reviewed
def compute_num_shifts(labels_: SER_ARR_TYPE) -> int:
    """
    Count the number of regime shifts in a (int) label array/series.
    """
    labels_arr = check_1d_array(labels_)
    return (labels_arr[:-1]!=labels_arr[1:]).sum()
