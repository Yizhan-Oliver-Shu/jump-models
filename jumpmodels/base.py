"""
Module for the base class used in clustering-like algorithms.

This module provides helpers for parameter sorting, parameter initialization, and base class 
definitions for clustering-like algorithms.

Depends on
----------
utils/ : Modules
"""

from .utils import *

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.cluster import kmeans_plusplus

##################################
# Sorting
##################################

# reviewed
def sort_param_dict_from_idx(params: dict, idx: np.ndarray) -> None:
    """
    Sort a dictionary of parameters according to a given index array.

    Expected parameter shapes:
    - `ret_` : (n_c,)
    - `vol_` : (n_c,)
    - `means_` : (n_c, n_f)
    - `centers_` : (n_c, n_f)
    - `transmat_` : (n_c, n_c)
    - `startprob_` : (n_c,)
    - `proba_` : (n_s, n_c)
    - `covars_` : (n_c, 1)

    Parameters
    ----------
    params : dict
        A dictionary of parameters, each corresponding to a clustering result.

    idx : ndarray of shape (n_c,)
        The index array to sort the parameters by.
    """
    # permute `axis=0`
    for key in ['ret_', 'vol_', 'means_', 'centers_', 'startprob_', 'covars_']:
        if key in params: params[key] = params[key][idx]
    # transmat, need to permute both `axis=0 & 1`
    if 'transmat_' in params: params['transmat_'] = params['transmat_'][idx][:, idx]
    # proba, need to permute `axis=1`
    if 'proba_' in params: params['proba_'] = params['proba_'][:, idx]
    return 
 
# reviewed
def sort_param_dict(params: dict, sort_by='ret') -> None:
    """
    Sort the states by a given criterion and permute all parameters accordingly.
    Supported sorting criteria are ["cumret", "vol", "freq", "ret"], i.e.
    states sorted by decreasing (cumulative) return, increasing vol, decreasing frequency.

    `nan` values will be (ideally) sorted to the end.

    Parameters
    ----------
    params : dict
        A dictionary of parameters, each corresponding to a clustering result.

    sort_by : str, optional (default='ret')
        The criterion to sort the parameters by. Must be one of ["cumret", "vol", "freq", "ret"].
    """
    if sort_by is None: return
    assert sort_by in ["cumret", "vol", "freq", "ret"]
    if "proba_" in params: freq = params["proba_"].sum(axis=0)
    if sort_by == 'vol':
        assert 'vol_' in params
        criterion = params['vol_']
    elif sort_by == "cumret":
        assert "ret_" in params and "proba_" in params
        criterion = -params["ret_"] * freq   # missing regimes will have a cumret of nan*0 = nan
    elif sort_by == "ret":
        assert "ret_" in params
        criterion = -params['ret_']
    elif sort_by == "freq":
        assert "proba_" in params
        criterion = -freq # decreasing freq
    else:
        raise NotImplementedError()
    criterion = replace_inf_by_nan(criterion)
    idx = np.argsort(criterion)  
    sort_param_dict_from_idx(params, idx)
    return 

# reviewed
def align_and_check_ret_ser(ret_ser: SER_ARR_TYPE, X: DF_ARR_TYPE) -> np.ndarray:
    """
    Align a return series with the input data matrix `X`,
    and convert it to a 1D array.

    Parameters
    ----------
    ret_ser : Series or ndarray
        The return series to validate.

    X : DataFrame or ndarray
        The data matrix to align with.

    Returns
    -------
    ndarray
        The aligned and validated 1D return array.
    """
    ret_ser = align_x_with_y(ret_ser, X)
    return check_1d_array(ret_ser)

# reviewed
def sort_states_from_ret(ret_ser: Optional[SER_ARR_TYPE], 
                         X: DF_ARR_TYPE,
                         best_res: dict, 
                         sort_by: str = "cumret") -> None:
    """
    Sort the states in the fitted parameters stored in a dictionary according to a specified criterion.
    This is intended for financial applications. If not applicable, input `None` for `ret_ser`.

    Parameters
    ----------
    ret_ser : Series or ndarray, optional
        The return series to use for computing average return and volatility within each state.
        If `None`, sorting is attempted by decreasing frequency (given that the `proba_` param is estimated).

    X : DataFrame or ndarray
        The data matrix to use for alignment.

    best_res : dict
        Fitted parameters of the best clustering results to sort.

    sort_by : str, optional (default="cumret")
        The criterion to use for sorting. Must be one of ["cumret", "vol", "freq", "ret"].

        - If `ret_ser` is provided, it is used to compute the mean return (`ret_`) and volatility (`vol_`) 
        within each state. Sorting by decreasing (cumulative) return and increasing volatility is possible. 
        - If `ret_ser` is `None`, sort by frequency if the `proba_` attribute exists, otherwise 
        don't sort anything.
    """
    if ret_ser is not None: 
        # valid inputs
        ret_ser_arr = align_and_check_ret_ser(ret_ser, X)
        # compute mean & vol for each cluster
        best_res['ret_'], best_res['vol_'] = weighted_mean_std_cluster(ret_ser_arr, best_res['proba_'])
        # the best parameters sorted by a criterion
        sort_param_dict(best_res, sort_by=sort_by)
    elif "proba_" in best_res:
        sort_param_dict(best_res, sort_by="freq")
    return 

##################################
# Initialization
##################################

# reviewed
def init_centers_kmeans_plusplus(X: np.ndarray, n_c=2, n_init=10, random_state=None) -> list[np.ndarray]:
    """
    Initialize the cluster centers using the K-Means++ algorithm, repeated `n_init` times.

    Parameters
    ----------
    X : ndarray of shape (n_s, n_f)
        The data matrix.

    n_c : int, optional (default=2)
        The number of clusters.

    n_init : int, optional (default=10)
        The number of initializations to perform.

    random_state : int, RandomState instance, or None, optional (default=None)
        Controls the randomness of the center initialization.

    Returns
    -------
    centers : list of ndarray
        A list of initialized centers for each run.
    """
    random_state = check_random_state(random_state)
    centers = [kmeans_plusplus(X, n_c, random_state=random_state)[0] for _ in range(n_init)]
    return centers   # (n_init, n_c, n_f)

##################################
# Base Class
##################################

class BaseClusteringAlgo(BaseEstimator):
    """
    A base class for all clustering-like algorithms.

    This class provides several common methods but does not include any model fitting logic. 
    It is intended to be inherited with specific implementations.

    Parameters
    ----------
    n_components : int
        The number of components (clusters).

    n_init : int
        The number of initializations to perform.

    max_iter : int
        The maximum number of iterations.

    tol : float
        The tolerance for convergence.

    random_state : int, RandomState instance, or None
        Controls the randomness.

    verbose : int
        Controls the verbosity of the output.
    """
    # reviewed
    def __init__(self,
                 n_components,
                 n_init,
                 max_iter,
                 tol,
                 random_state,
                 verbose
                 ) -> None:
        self.n_components = n_components
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

    # reviewed
    def is_shape_match_X_centers(self, X: DF_ARR_TYPE) -> bool:
        """
        Check whether the shape of `X` and `centers_` matches. Useful for `predict` methods.
        `self` must already has the attribute `centers_`.

        Parameters
        ----------
        X : DataFrame or ndarray
            The input data matrix.

        Returns
        -------
        bool
            True if the shapes match, False otherwise.
        """
        n_f = X.shape[1]
        return self.centers_.shape == (self.n_components, n_f)
    
    # reviewed
    def init_centers(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize the centers using k-Means++ for multiple initializations. 
        If attribute `centers_` exists and matches the shape of `X`, it will also 
        be included as an initial value.

        Parameters
        ----------
        X : ndarray of shape (n_s, n_f)
            The input data matrix.

        Returns
        -------
        centers : ndarray
            The initialized centers for each run.
        """
        centers = init_centers_kmeans_plusplus(X, self.n_components, self.n_init, self.random_state)
        if hasattr(self, "centers_") and self.is_shape_match_X_centers(X): 
            centers.append(self.centers_)  # use previously fitted value as one initial center value
        return np.array(centers)
    
    # reviewed
    def check_X_predict_func(self, X: DF_ARR_TYPE) -> np.ndarray:
        """
        Check the input data matrix for `.predict` methods, ensuring it is a 2D array and 
        matches the shape of `centers_`.

        Parameters
        ----------
        X : DataFrame or ndarray
            The input data matrix.

        Returns
        -------
        ndarray
            The validated 2D data array.
        """
        X_arr = check_2d_array(X)
        assert self.is_shape_match_X_centers(X_arr)
        return X_arr
 