"""
utils of basic numerical calculation.
"""

from .validation import *

# will not raise warnings if: divide by zero, take sqrt of nega values
np.seterr(divide="ignore", invalid="ignore")

# reviewed
def np_divide(x, y):
    """
    np.divide(). if divide by zero, don't raise warning, and return np.nan (instead of np.inf)
    """
    res = np.divide(x, y)
    if not np.isinf(res).any(): return res  # no inf
    if np.isscalar(res): return np.nan  # single inf scalar
    # replace inf with nan
    res[np.isinf(res)] = np.nan
    return res

# reviewed
def normalize_weight(w: SER_ARR_TYPE) -> SER_ARR_TYPE:
    """
    divide a vector by its sum. sum must be positive
    """
    w_sum = w.sum()
    assert w_sum > 0
    return w/w_sum

#################################
## weighted ave
#################################
    
# reviewed
def fill_na(X: np.ndarray, fill_na_arr: Union[None, np.ndarray] = None) -> np.ndarray:
    """
    fill the nans in a 1-d/2-d arr X by values in `fill_na_arr`.

    Three situations:
    - `fill_na_arr` is a scalar number, then fill in every missing
    - `fill_na_arr` and `X` have the same `ndim`, then their shapes must be the same, and values from `fill_na_arr` will be filled.
    -  X.ndim==2 and fill_na_arr.ndim == 1. 
        nans in `X` must occupy whole rows (an entire missing row), and `fill_na_arr` will be filled in these missing rows.
    """
    idx = np.isnan(X)
    if not idx.any() or fill_na_arr is None: # no nans or no values provided
        return X
    # fill na values
    X = X.copy()
    assert X.ndim in [1, 2]
    if is_numbers(fill_na_arr) or fill_na_arr.ndim == 0:    # scalar number
        X[idx] = fill_na_arr
    elif X.ndim == fill_na_arr.ndim:  # same shape, ndim == 1/2
        assert X.shape == fill_na_arr.shape
        X[idx] = fill_na_arr[idx]
    else:   # X.ndim==2 and fill_na_arr.ndim == 1
        assert fill_na_arr.shape == (X.shape[1],)
        assert is_rows_all_True_or_all_False(idx)
        X[idx[:, 0]] = fill_na_arr    # missing rows
    return X

# reviewed
def weighted_mean_cluster(X: np.ndarray, weights: np.ndarray, fill_na_mean=None) -> np.ndarray:
    """
    compute the weighted sample average for each cluster. X should be an arr. if total weights sum to 0 (no observation), could fill in the values in `fill_na_arr`.

    Parameters:
    -----------------------------
    X: arr (n_s,) or (n_s, n_f)
        the data matrix.
    weights: arr (n_s, n_c)
        weight arr. must be all nonnegative. could support shape of (n_s,) later if needed.
    fill_na_mean:
        values for fill for mean if NaNs exist.
    """
    # valide inputs
    assert X.ndim in [1, 2]   # (n_s,) or (n_s, n_f)
    assert (weights >= 0).all()
    assert weights.ndim == 2    # (n_s, n_c). support 1d weights later?
    assert len(X) == len(weights)
    # 
    weighted_sum = weights.T @ X        # (n_c,) or (n_c, n_f)
    Ns = weights.sum(axis=0)   # (n_c, )
    if X.ndim == 2: Ns = Ns[:, np.newaxis]  # for correct broadcasting
    average = np_divide(weighted_sum, Ns)
    average = fill_na(average, fill_na_mean)
    return average

# reviewed
def weighted_mean_std_cluster(X: np.ndarray, weights: np.ndarray, fill_na_mean=None, fill_na_std=None, bias=False) -> np.ndarray:
    """
    compute the weighted means and stds for each cluster

    Parameters:
    -----------------------------
    X: arr (n_s,) or (n_s, n_f)
        the data matrix.
    weights: arr (n_s, n_c)
        weight arr. must be all nonnegative. could support shape of (n_s,) later if needed.
    fill_na_mean:
        values for fill for mean if NaNs exist.
    fill_na_std:
        values for fill for std if NaNs exist.
    bias: bool
        use biased or unbiased estimator for std.
    """
    means_ = weighted_mean_cluster(X, weights, fill_na_mean=fill_na_mean)     # (n_c,) / (n_c, n_f)
    weighted_sq = weighted_mean_cluster(X ** 2, weights, fill_na_mean=None) - means_ ** 2      # (n_c,) / (n_c, n_f)
    if not bias:    # debiase factor
        factor = np_divide(weights.sum(axis=0) ** 2, weights.sum(axis=0) ** 2 - (weights**2).sum(axis=0))     # (n_c,)
        if weighted_sq.ndim == 2: factor = factor[:, np.newaxis]     # if weighted_sq (n_c, n_f), then factor (n_c, 1)
        weighted_sq *= factor
    stds_ = np.sqrt(weighted_sq)  # (n_c,) / (n_c, n_f)
    stds_ = fill_na(stds_, fill_na_std)
    return means_, stds_