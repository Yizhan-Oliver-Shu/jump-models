"""
clustering-related utils.
"""

import numpy as np
from .validation import *
from .calculation import *

# reviewed
def is_valid_labels(labels_: SER_ARR_TYPE, n_c=2):
    """
    return whether a labels arr/ser is a valid label seq.
    `labels_` can be of float/int dtype, but its values must lie in the set of {0, 1, .., n_c-1}.
    """
    labels_arr = check_1d_array(labels_)   # check whether it is intrinsically 1-d
    return set(labels_arr).issubset(set(range(n_c)))

# reviewed
def is_valid_proba(proba_: DF_ARR_TYPE):
    """
    return whether a proba arr/ser is a valid one, e.g. all non-nega & all rows sum to 1.
    """
    proba_arr = check_2d_array(proba_)
    return (proba_arr>=0).all() and np.isclose(proba_arr.sum(axis=1), 1.).all()

# reviewed
def raise_labels_into_proba(labels_: np.ndarray, n_c: int) -> np.ndarray:
    """
    turns a discrete labels array into a probability mx. 
    labels must be of int type, and have value smaller than `n_c` (won't check them for efficiency)
    
    Parameters:
    -----------------------
    labels_: array of ints, (n_s,)

    n_c: int


    Returns:
    ------------------------
    proba_: a probability assignment array with 0./1. values, (n_s, n_c). correspond to the hard clustering result `labels_`
    """
    # labels_ must be ints, and smaller than n_c
    # don't check input labels, as output proba_ will be checked
    n_s = len(labels_)
    proba_ = np.zeros((n_s, n_c)) 
    proba_[range(n_s), labels_] = 1.
    assert is_valid_proba(proba_)
    return proba_

# reviewed
def reduce_proba_to_labels(proba_: DF_ARR_TYPE) -> SER_ARR_TYPE:
    """
    convert a proba mx into a label series, by taking the argmax of every row.
    treat differently with df versus arr.
    """
    if is_ser_df(proba_): return proba_.idxmax(axis=1)
    return proba_.argmax(axis=1).astype(np.int32)

# reviewed
def is_map_from_left_to_right(labels_left: np.ndarray, labels_right: np.ndarray) -> bool:
    """
    check whether the map from the left labels to the right is indeed a map, i.e. elements with the same label in `labels_left` should also be the same in `labels_right`.
    if either labels is None, return False
    """
    if labels_left is None or labels_right is None:
        return False
    assert len(labels_left) == len(labels_right)
    for label in np.unique(labels_left):
        if len(np.unique(labels_right[labels_left==label])) != 1:
            return False
    return True

# reviewed
def is_same_clustering(labels1: np.ndarray, labels2: np.ndarray) -> bool:
    """
    check whether two clustering results are the same, under permutation.
    not the same result if the unique labels of the two results don't match.
    
    if either input is None, return False
    """
    return is_map_from_left_to_right(labels1, labels2) and is_map_from_left_to_right(labels2, labels1)

# reviewed
def empirical_trans_mx(labels_: SER_ARR_TYPE, n_components=2, return_counts=False):
    """
    compute the empirical transition counts / prob mx from a labels arr/ser.
    
    labels must be in {0, 1, ...,  K-1}, but can be both float/int type.
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
    return np_divide(count_mx, count_mx.sum(axis=1, keepdims=True))     # will return nan if a state never appears.

# reviewed
def compute_num_shifts(labels_: SER_ARR_TYPE) -> int:
    """
    count how many regime shifts, for a int labels ser/arr
    """
    labels_arr = check_1d_array(labels_)
    return (labels_arr[:-1]!=labels_arr[1:]).sum()