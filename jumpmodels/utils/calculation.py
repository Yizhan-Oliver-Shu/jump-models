"""
Helpers for basic numerical calculations.

This module focuses on numerical calculations with special attention to `numpy` behaviors 
involving NaN and infinity:

- 0. / 0. = np.nan
- 0. * np.inf = np.nan
- 0. * np.nan = np.nan
- 1. / 0. = np.inf
- -1. / 0. = -np.inf

Typically, it is rare for a statement to directly yield `np.inf`; the first two examples 
are the most common cases.

Depends on
----------
utils.validation : Module
"""

from .validation import *

# will not raise warnings if: divide by zero, take sqrt of nega values
np.seterr(divide="ignore", invalid="ignore")

# reviewed
def set_zero_arr(x: np.ndarray, tol=1e-6) -> np.ndarray:
    """
    Set elements of a numpy array that are close to zero to exactly zero.

    Parameters
    ----------
    x : ndarray
        The input numpy array.

    tol : float, optional (default=1e-6)
        The tolerance value. Elements with absolute values smaller than `tol` 
        are set to zero.

    Returns
    -------
    ndarray
        A numpy array with near-zero values replaced by exact zeros.
    """
    return np.where(np.abs(x) < tol, 0., x)

# reviewed
def replace_inf_by_nan(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Replace both positive and negative infinity values with NaN in a float or numpy array.

    Parameters
    ----------
    x : float or ndarray
        The input float or numpy array.

    Returns
    -------
    float or ndarray
        A float or numpy array with infinities replaced by NaN.
    """
    return np.where(np.isinf(x), np.nan, x)

# reviewed
def replace_nan_by_inf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Replace all NaN values with positive infinity in a float or numpy array.

    Parameters
    ----------
    x : float or ndarray
        The input float or numpy array.

    Returns
    -------
    float or ndarray
        A float or numpy array with NaN values replaced by infinity.
    """
    return np.where(np.isnan(x), np.inf, x)

# reviewed
def decre_verbose(verbose: int) -> int:
    """
    Decrement a non-negative integer by 1, ensuring the result is non-negative.

    Parameters
    ----------
    verbose : int
        A non-negative integer to decrement.

    Returns
    -------
    int
        The decremented value, ensuring it is non-negative.
    """
    return max(0, verbose-1)

#################################
## weighted ave
#################################

# reviewed
def weighted_mean_cluster(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Compute the weighted sample average for each cluster. `X` can be a 1D or 2D array.
    If the total weights sum to zero (indicating no observation), return `np.nan`.
    No `np.inf` will appear in the result.

    Parameters
    ----------
    X : ndarray of shape (n_s,) or (n_s, n_f)
        The data matrix, where `n_s` is the number of samples and `n_f` is the number of features.

    weights : ndarray of shape (n_s, n_c)
        The weight array for each sample and cluster. Must be all non-negative. Support for 
        `weights` of shape (n_s,) can be added later if needed.

    Returns
    -------
    ndarray of shape (n_c,) or (n_c, n_f)
        The weighted mean for each cluster.
    """
    # valid X
    assert X.ndim in [1, 2]   # (n_s,) or (n_s, n_f)
    X_2d = check_2d_array(X, assert_na=False)   # (n_s, n_f)
    # valid weights
    weights = check_2d_array(weights, assert_na=False)   # (n_s, n_c)
    assert len(X_2d) == len(weights)
    assert (weights >= 0).all()
    # 
    weighted_sum = weights.T @ X_2d        # (n_c, n_f)
    Ns = weights.sum(axis=0, keepdims=True).T   # (n_c, 1)
    means_ = weighted_sum / Ns   # (n_c, n_f)
    if X.ndim == 1: means_ = means_.squeeze()
    return means_        # (n_c,) or (n_c, n_f)

# reviewed
def weighted_mean_std_cluster(X: np.ndarray, weights: np.ndarray, bias=False) -> np.ndarray:
    """
    Compute the weighted means and standard deviations for each cluster.

    In extreme cases leading to NaNs (otherwise, all values are normal):
    - No observation: both `var_` and `factor` will be NaNs, and standard deviation will also be NaN.
    - Only one observation: `var_` will be zero, while `factor` will be `np.inf`. When considering the debiasing 
      factor, this results in NaN standard deviations.

    Parameters
    ----------
    X : ndarray of shape (n_s,) or (n_s, n_f)
        The data matrix, where `n_s` is the number of samples and `n_f` is the number of features.

    weights : ndarray of shape (n_s, n_c)
        The weight array for each sample and cluster. Must be all non-negative.

    bias : bool, optional (default=False)
        If False, apply a debiasing factor to the variance calculation.

    Returns
    -------
    means_ : ndarray of shape (n_c,) or (n_c, n_f)
        The weighted mean for each cluster.

    stds_ : ndarray of shape (n_c,) or (n_c, n_f)
        The weighted standard deviation for each cluster.
    """
    X_2d = check_2d_array(X, assert_na=False)    # (n_s, n_f)
    means_ = weighted_mean_cluster(X_2d, weights)   # (n_c, n_f)
    sq_means_ = weighted_mean_cluster(X_2d ** 2, weights)   # (n_c, n_f)
    var_ = sq_means_ - means_ ** 2  # (n_c, n_f)
    if not bias:    # debiase factor, see: https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Reliability_weights
        V1 = weights.sum(axis=0, keepdims=True)     # (1, n_c)
        V2 = (weights**2).sum(axis=0, keepdims=True)   # (1, n_c)
        factor = 1. / (1. - V2/V1**2)   # (1, n_c)
        factor = factor.T   # (n_c, 1)
        var_ *= factor  # (n_c, n_f)
    stds_ = np.sqrt(var_)  # (n_c, n_f)
    if X.ndim == 1:
        return means_.squeeze(), stds_.squeeze()
    return means_, stds_
