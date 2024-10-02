"""
Module for Sparse Jump Models (SJMs).

This module provides an implementation of sparse jump models, extending the jump model 
with additional support for feature selection through Lasso-like optimization.

Depends on
----------
utils/ : Modules
    Utility functions for validation and clustering operations.
jump : Module
    Discrete and continuous jump models.
"""

from .utils import *
from .jump import *

from numpy.linalg import norm

########################################################
## Lasso Problem for Feature Weights
########################################################

# reviewed
def binary_search_decrease(func, 
                           left: float, 
                           right: float, 
                           value: float, 
                           *args, 
                           tol_x: float = 1e-8, 
                           tol_y: float = 0., 
                           max_iter: int = 100, 
                           verbose: int = 0,
                           **kwargs) -> float:
    """
    Binary search for a decreasing function.

    This method performs binary search to find the point where the function `func` 
    decreases to a specified value within given tolerances.

    Parameters
    ----------
    func : callable
        The function to be minimized.
    
    left : float
        The left bound for the search.

    right : float
        The right bound for the search.

    value : float
        The target value to find.

    tol_x : float, optional (default=1e-8)
        The tolerance for the search along the x-axis.

    tol_y : float, optional (default=0.)
        The tolerance for the search along the y-axis (function value).

    max_iter : int, optional (default=100)
        Maximum number of iterations.

    verbose : int, optional (default=0)
        Verbosity level. If greater than 0, prints progress information.

    Returns
    -------
    float
        The optimal point where the function reaches the target value.
    """
    if value >= func(left): return left
    if value <= func(right): return right
    # 
    gap = right-left
    num_iter = 0
    while (gap > tol_x and num_iter < max_iter):
        # print(f"{left}, {right}")
        num_iter += 1
        middle = (right + left) / 2
        func_call = func(middle, *args, **kwargs)
        if verbose: print("x value", middle, "y value", func_call)
        if func_call < value-tol_y/2:
            right = middle
        elif func_call > value+tol_y/2:
            left = middle
        else:
            return middle
        gap /= 2
    if num_iter < max_iter:
        return middle
    raise Exception("Non-convergence: Possible mathematical error.")
    
# reviewed
def soft_thres_l2_normalized(x: SER_ARR_TYPE, thres: float = 0.) -> SER_ARR_TYPE:
    """
    Soft thresholding for a non-negative vector `x`, followed by L2 normalization.

    Parameters
    ----------
    x : Series or ndarray
        The input vector to be thresholded and normalized.

    thres : float, optional (default=0.)
        The threshold for soft thresholding.

    Returns
    -------
    Series or ndarray
        The thresholded and L2-normalized vector.
    """
    y = np.maximum(0, x-thres)
    y_norm = norm(y)
    assert y_norm > 0
    return y / y_norm

# reviewed
def solve_lasso(a: SER_ARR_TYPE, 
                norm_ub: float, 
                tol: float = 1e-8) -> SER_ARR_TYPE:
    """
    Solve the Lasso problem for feature weights.

    This function finds the optimal feature weights subject to the constraint that the 
    L1-norm of the weights is bounded by `norm_ub`.

    Parameters
    ----------
    a : Series or ndarray
        The input vector for the Lasso problem.

    norm_ub : float
        The upper bound for the L1-norm of the feature weights. 
        Equals to `kappa` in the published articles.

    tol : float, optional (default=1e-8)
        The tolerance for the binary search.

    Returns
    -------
    Series or ndarray
        The optimized feature weights.
    """
    assert norm_ub >= 1.
    a_arr = check_1d_array(a)
    left, right = 0., np.unique(a_arr)[-2]  # right is the second largest element of `a`
    if right < tol: thres_sol = 0.
    else:
        func = lambda thres: soft_thres_l2_normalized(a_arr, thres).sum()
        thres_sol = binary_search_decrease(func, left, right, norm_ub, tol_x=tol)
    # return thres_sol
    w = soft_thres_l2_normalized(a_arr, thres_sol)
    return raise_arr_to_pd_obj(w, a)

# reviewed
def compute_BCSS(X: DF_ARR_TYPE, 
                 proba_: DF_ARR_TYPE, 
                 centers_: Optional[np.ndarray] = None,
                 tol: float = 1e-6) -> SER_ARR_TYPE:
    """
    Compute the Between Cluster Sum of Squares (BCSS).

    The BCSS is computed based on the cluster centers and probabilities. If no centers are provided, 
    they will be computed from probabilities. Any BCSS values below the tolerance are set to zero.

    Parameters
    ----------
    X : DataFrame or ndarray
        The input data matrix.

    proba_ : DataFrame or ndarray
        The cluster assignment probabilities.

    centers_ : ndarray, optional
        The cluster centers. NA values are acceptable.
        If not provided, they are estimated from the data.

    tol : float, optional (default=1e-6)
        The tolerance for setting BCSS values to zero.

    Returns
    -------
    Series or ndarray
        The BCSS values for each feature.
    """
    X_arr, proba_arr = check_2d_array(X), check_2d_array(proba_)
    if centers_ is None: centers_ = weighted_mean_cluster(X_arr, proba_arr)
    # replace NAs in centers with 0. won't affect computation
    centers_ = np.nan_to_num(centers_, nan=0.)
    # assert not np.isnan(centers_).any()
    Ns = proba_arr.sum(axis=0)
    BCSS = Ns @ ((centers_ - X_arr.mean(axis=0))**2)
    BCSS = set_zero_arr(BCSS, tol=tol)
    assert not np.isnan(BCSS).any()
    return raise_arr_to_pd_obj(BCSS, X, index_key="columns")

############################
## SJM
############################

class SparseJumpModel(BaseEstimator):
    """
    Sparse Jump Model (SJM) with feature selection.

    This model extends the standard jump model by incorporating a Lasso-like feature 
    selection process, where the number of selected features is controlled by `max_feats`.

    Parameters
    ----------
    n_components : int, default=2
        Number of components (clusters).

    max_feats : float, default=100.
        Controls the number of features included. This is the square of `kappa`, and 
        represents the effective number of features.

    jump_penalty : float, default=0.
        The jump penalty. In SJM, this penalty is scaled by 
        `1 / sqrt(n_features)` since features are weighted.

    cont : bool, default=False
        If `True`, the continuous jump model is used. Otherwise, the discrete model is applied.

    grid_size : float, default=0.05
        The grid size for discretizing the probability simplex (only used for continuous models).

    mode_loss : bool, default=True
        Whether to apply the mode loss penalty (only relevant for continuous models).

    random_state : int or RandomState, optional
        Random number generator seed for reproducibility.

    max_iter : int, default=30
        Maximum number of iterations for the coordinate descent algorithm in feature selection.

    tol_w : float, default=1e-4
        Tolerance for stopping the optimization of feature weights.

    max_iter_jm : int, default=1000
        Maximum number of iterations for the jump model fitting process.

    tol_jm : float, default=1e-8
        Stopping tolerance for the jump model fitting.

    n_init_jm : int, default=20
        Number of initializations for the jump model.

    verbose : int, default=0
        Controls the verbosity of the output.

    Attributes
    ----------
    jm_ins : JumpModel
        The fitted jump model instance, with feature weighting.

    feat_weights : ndarray
        The optimal feature weights.
        Square root of the `w` vector in the oroginal SJM formulation.

    labels_ : Series or ndarray
        In-sample optimal state assignments.

    proba_ : DataFrame or ndarray
        In-sample optimal probability matrix.

    ret_, vol_ : Series or ndarray
        Average return (`ret_`) and volatility (`vol_`) for each state, if `ret_ser` is provided.

    centers_ : ndarray
        The weighted cluster centers.
    """
    # reviewed
    def __init__(self,
                 n_components: int = 2, 
                 max_feats: float = 100.,
                 jump_penalty: float = 0., 
                 cont: bool = False, 
                 grid_size: float = 0.05, 
                 mode_loss: bool = True, 
                 random_state = RANDOM_STATE, 
                 max_iter: int = 30, 
                 tol_w: float = 1e-4, 
                 max_iter_jm: int = 1000,
                 tol_jm: float = 1e-8,
                 n_init_jm: int = 20,
                 verbose: int = 0):
        self.n_components = int(n_components)
        self.max_feats = max_feats
        self.jump_penalty = jump_penalty
        self.cont = cont
        self.grid_size = grid_size
        self.mode_loss = mode_loss
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol_w = tol_w
        self.max_iter_jm = max_iter_jm
        self.tol_jm = tol_jm
        self.n_init_jm = n_init_jm
        self.verbose = verbose

    # reviewed
    def init_jm(self):
        """
        Initialize the jump model instance with scaled jump penalty.
        """
        jump_penalty = self.jump_penalty / np.sqrt(self.n_features_all)
        jm = JumpModel(n_components=self.n_components,
                       jump_penalty=jump_penalty,
                       cont=self.cont,
                       grid_size=self.grid_size,
                       mode_loss=self.mode_loss,
                       random_state=self.random_state,
                       max_iter=self.max_iter_jm,
                       tol=self.tol_jm,
                       n_init=self.n_init_jm,
                       verbose=decre_verbose(self.verbose))
        self.jm_ins = jm
        return jm
    
    # reviewed
    def print_log(self, n_iter, BCSS, w):
        """
        Print fitting logs if verbosity is enabled.
        """
        if self.verbose:
            print("Iter:", n_iter)
            print("BCSS:\n", BCSS)     #, "sum:", BCSS.sum()
            print("w:\n", w, "\n")
        return 

    # reviewed
    def fit(self, 
            X: DF_ARR_TYPE, 
            ret_ser: Optional[SER_ARR_TYPE] = None,
            sort_by: Optional[str] = "cumret"):
        """
        Fit the sparse jump model using coordinate descent.

        This method iteratively optimizes the feature weights and fits the jump model 
        on the weighted data.

        Parameters
        ----------
        X : DataFrame or ndarray
            The input data matrix.

        ret_ser : Series or ndarray, optional
            A return series used for sorting states.

        sort_by : ["cumret", "vol", "freq", "ret"], optional (default="cumret")
            Criterion for sorting states.

        Returns
        -------
        SparseJumpModel
            The fitted sparse jump model.
        """
        #
        X_arr = check_2d_array(X)
        self.n_features_all = X_arr.shape[1]
        # jm ins
        jm = self.init_jm()
        # get attrs
        max_iter = self.max_iter
        tol_w = self.tol_w
        norm_ub = np.sqrt(self.max_feats)
        # 
        w_old = np.ones(self.n_features_all)*2  # not a valid weight, only used for entering the 1st iter
        w = np.ones(self.n_features_all) / np.sqrt(self.n_features_all)  # initial weight   #  np.repeat(1/np.sqrt(self.n_features_all), self.n_features_all)  
        n_iter = 0
        while (n_iter < max_iter and norm(w-w_old, 1) / norm(w_old, 1) > tol_w):
            # 
            n_iter += 1
            w_old = w
            # Step 1: fix w, fit JM
            feat_weights = np.sqrt(w)
            # use the previous optimal center, weighted by the most recent w, as an initialization
            if n_iter > 1: jm.centers_ = centers_unweighted * feat_weights    
            # fit JM on weighted data
            jm.fit(X, ret_ser=ret_ser, feat_weights=feat_weights, sort_by=sort_by)
            # Step 2: optimize w
            # update (unweighted) centers
            centers_unweighted = weighted_mean_cluster(X_arr, jm.proba_)
            # compute BCSS on the original data
            BCSS = compute_BCSS(X_arr, jm.proba_, centers_unweighted)
            if (BCSS <= 0).all(): # all in one cluster
                self.print_log(n_iter, BCSS, w)
                break
            w = solve_lasso(BCSS/BCSS.max(), norm_ub)
            self.print_log(n_iter, BCSS, w)
        # best res
        self.w = raise_arr_to_pd_obj(w, X, index_key="columns")
        self.feat_weights = raise_arr_to_pd_obj(jm.feat_weights, X, index_key="columns")
        self.centers_ = jm.centers_ # weighted centers
        # self.centers_ = weighted_mean_cluster(X_arr, jm.proba_, )
        self.labels_ = jm.labels_
        self.proba_ = jm.proba_
        if ret_ser is not None:
            self.ret_ = jm.ret_
            self.vol_ = jm.vol_
        return self
    
    def predict_proba_online(self, X: DF_ARR_TYPE) -> DF_ARR_TYPE:
        """
        Predict state probabilities in an online fashion.
        """
        return self.jm_ins.predict_proba_online(X)
    
    def predict_online(self, X: DF_ARR_TYPE) -> SER_ARR_TYPE:
        """
        Predict states in an online fashion.
        """
        return self.jm_ins.predict_online(X)
    
    def predict_proba(self, X: DF_ARR_TYPE) -> DF_ARR_TYPE:
        """
        Predict state probabilities using all available data.
        """
        return self.jm_ins.predict_proba(X)

    def predict(self, X: DF_ARR_TYPE) -> SER_ARR_TYPE:
        """
        Predict states using all available data.
        """
        return self.jm_ins.predict(X)