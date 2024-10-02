"""
Module for statistical jump models (JMs) and continuous jump models (CJMs).

This module provides utilities and helper functions for implementing and working 
with jump models and their continuous variants.

Depends on
----------
utils/ : Modules
    Utility functions for validation and clustering operations.
base : Module
    Base class for clustering-like algorithms.
"""

from itertools import product
from scipy.spatial.distance import cdist
from scipy.special import logsumexp

from . import RANDOM_STATE
from .utils import *
from .base import *

#################################
## model helpers
#################################

# reviewed
def jump_penalty_to_mx(jump_penalty: float, n_c: int) -> np.ndarray:
    """
    Convert a scalar jump penalty into a penalty matrix.

    Parameters
    ----------
    jump_penalty : float
        The scalar value representing the jump penalty.

    n_c : int
        The number of clusters or components.

    Returns
    -------
    np.ndarray
        A matrix of shape (n_c, n_c) where off-diagonal elements are the penalty values 
        and diagonal elements are zero.
    """
    # assert is_numbers(jump_penalty)
    return jump_penalty * (np.ones((n_c, n_c)) - np.eye(n_c))   # default dtype is float

# reviewed
def discretize_prob_simplex(n_c: int, grid_size: float) -> np.ndarray:
    """
    Sample grid points on a probability simplex. This function generates all possible 
    combinations of probabilities that sum to 1, given the grid size.
    NB: this operation is of combinatorial complexity.

    Parameters
    ----------
    n_c : int
        The number of components or clusters.

    grid_size : float
        The step size for discretization of the simplex.

    Returns
    -------
    np.ndarray
        An array of shape (n_candidates, n_c), where each row represents a point on the 
        simplex. The number of candidates depends on the grid size.
    """
    N = int(1/grid_size)
    tuples = filter(lambda x: sum(x)==N, product(range(N+1), repeat = n_c))
    lst = np.array(list(tuples)[::-1], dtype=float)/N   # (n_candidates, n_c)
    return lst

#################################
## DP algo & E step
#################################

# reviewed
def dp(loss_mx: np.ndarray, 
       penalty_mx: np.ndarray, 
       return_value_mx: bool = False) -> Union[tuple[np.ndarray, float], np.ndarray]:
    r"""
    Solve the optimization problem involved in the E-step calculation (state assignment), 
    using a dynamic programming (DP) algorithm.

    The objective is to minimize:

    $$\min \sum_{t=0}^{T-1} L(t, s_t) + \sum_{t=1}^{T-1} \Lambda(s_{t-1}, s_t).$$

    If some columns of `loss_mx` contain `NaN` values, they are replaced with `inf`, 
    making those clusters unreachable.

    Note: The DP algorithm cannot be easily sped up using Numba due to issues with 
    `.min(axis=0)` in Numba.

    Parameters
    ----------
    loss_mx : ndarray of shape (n_s, n_c)
        The loss matrix, where `L(t, k)` represents the loss for time `t` and state `k`.

    penalty_mx : ndarray of shape (n_c, n_c)
        The jump penalty matrix between states.

    return_value_mx : bool, optional (default=False)
        If `True`, compute and return the value matrix from the DP algorithm. The value at 
        each time step `t` is based on all information up to that point, making it suitable 
        for online inference.

    Returns
    -------
    tuple[np.ndarray, float] or np.ndarray
        If `return_value_mx` is `False`, returns a tuple containing:
        - The optimal state assignments.
        - The optimal loss function value.
        
        If `return_value_mx` is `True`, returns the value matrix.
    """
    # valid shape
    n_s, n_c = loss_mx.shape
    assert penalty_mx.shape == (n_c, n_c)
    # replace nan by inf
    loss_mx = replace_nan_by_inf(loss_mx)
    # DP algo
    values, assign = np.empty((n_s, n_c)), np.empty(n_s, dtype=int)
    # initial
    values[0] = loss_mx[0]
    # DP iteration
    for t in range(1, n_s):
        values[t] = loss_mx[t] + (values[t-1][:, np.newaxis] + penalty_mx).min(axis=0) # values[t-1][:, np.newaxis] turns the (t-1)-th row into a column
    # 
    if return_value_mx:
        return values
    # find optimal path backwards
    assign[-1] = values[-1].argmin()
    value_opt = values[-1, assign[-1]]
    # traceback
    for t in range(n_s - 1, 0, -1):
        assign[t-1] = (values[t-1] + penalty_mx[:, assign[t]]).argmin()
    return assign, value_opt

# reviewed
def raise_JM_labels_to_proba(labels_: np.ndarray, n_c: int, prob_vecs: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Convert JM labels into a probability matrix. If `prob_vecs` is provided, 
    the probability matrix is constructed using the probability vectors corresponding to each label. 
    Otherwise, a hard-clustering probability matrix is created from the labels.
    """
    return prob_vecs[labels_] if prob_vecs is not None else raise_labels_into_proba(labels_, n_c)

# reviewed
def raise_JM_proba_to_df(proba_: np.ndarray, X: DF_ARR_TYPE) -> DF_ARR_TYPE:
    """
    Convert a probability matrix into a pandas DataFrame, aligning with the index of the input 
    data matrix `X`.
    """
    return raise_arr_to_pd_obj(proba_, X, columns_key=None, return_as_ser=False)

LARGE_FLOAT = 1e100

# reviewed
def do_E_step(X: np.ndarray, 
              centers_: np.ndarray, 
              penalty_mx: np.ndarray, 
              prob_vecs: Optional[np.ndarray] = None, 
              return_value_mx: bool = False) -> Union[tuple[np.ndarray, np.ndarray, float], np.ndarray]:
    """
    Perform a single E-step: compute the loss matrix and calling the solver.

    This function handles both hard clustering and continuous models. The `centers_` parameter 
    can contain `NaN` values. It returns the probabilities, labels, and optimal value, where 
    `labels_` correspond to the state space.

    Parameters
    ----------
    X : ndarray of shape (n_s, n_f)
        The input data matrix, where `n_s` is the number of samples and `n_f` is the number of features.

    centers_ : ndarray of shape (n_c, n_f)
        The cluster centers. Can contain `NaN` values.

    penalty_mx : ndarray of shape (n_c, n_c)
        The penalty matrix representing the transition cost between states.

    prob_vecs : ndarray of shape (N, n_c), optional
        Probability vectors for the continuous model. If provided, this adjusts the loss matrix.

    return_value_mx : bool, optional (default=False)
        If `True`, return the value matrix from the DP algorithm, which can be used for online inference.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float] or np.ndarray
        If `return_value_mx` is `False`, returns a tuple containing:
        - `proba_` : ndarray of shape (n_s, n_c)
            The probability matrix, where each row corresponds to the probabilities for a sample.
        - `labels_` : ndarray of shape (n_s,)
            The state labels assigned to each sample.
        - `val_` : float
            The optimal value of the objective function.

        If `return_value_mx` is `True`, returns the value matrix instead of the tuple.
    """
    n_c = len(centers_)     # (n_c, n_f)
    # compute loss matrix
    loss_mx = .5 * cdist(X, centers_, "sqeuclidean")    # (n_s, n_c)
                                                        # contain `nan` if `centers_` contains `nan`.
    if prob_vecs is not None:    # cont model, (N, n_c)
        # replace the nan in loss_mx by a very large floating number
        loss_mx = np.nan_to_num(loss_mx, nan=LARGE_FLOAT, posinf=LARGE_FLOAT, neginf=LARGE_FLOAT)
        loss_mx = loss_mx @ prob_vecs.T     # each pair of loss between period t and candidate vector, (n_s, N)
    if return_value_mx: return dp(loss_mx, penalty_mx, return_value_mx=True)
    # do a full E step
    labels_, val_ = dp(loss_mx, penalty_mx, return_value_mx=False)     # output labels_ is of type int
    proba_ = raise_JM_labels_to_proba(labels_, n_c, prob_vecs)
    return proba_, labels_, val_    # the returned proba_ must be a valid proba arr

#################################
## feature weights
#################################

# reviewed
def valid_feat_weights(feat_weights: Optional[SER_ARR_TYPE]) -> None:
    """
    Validate the input `feat_weights`, ensuring all weights are non-negative and at least 
    one is positive. This function is called at the beginning of the method to ensure 
    the feature weights are valid.

    Parameters
    ----------
    feat_weights : Series or ndarray, optional
        The array of feature weights to validate. If `None`, no validation is performed.

    Raises
    ------
    AssertionError
        If any feature weights are negative or if no positive weights exist.
    """
    if feat_weights is None: return 
    feat_weights_arr = check_1d_array(feat_weights)
    assert (feat_weights_arr >= 0.).all(), "Feature weights must be non-negative."
    assert (feat_weights_arr > 0.).any(), "At least one feature weight must be positive."
    return 

# reviewed
def _valid_shape_X_feat_weights(X: DF_ARR_TYPE, feat_weights: Optional[SER_ARR_TYPE]) -> None:
    """
    Assert that the dimensions of the input data matrix `X` and feature weights match.

    Parameters
    ----------
    X : DataFrame or ndarray
        The input data matrix.

    feat_weights : Series or ndarray, optional
        The array of feature weights. If `None`, no assertion is made.

    Raises
    ------
    AssertionError
        If the dimensions of `X` and `feat_weights` do not match.
    """
    if feat_weights is None: 
        return
    if is_ser_df(X) and is_ser_df(feat_weights): 
        assert (X.columns==feat_weights.index).all(), "Feature mismatch: column names do not match feature weight index."
    else: 
        assert X.shape[1]==len(feat_weights) , "Feature mismatch: number of features does not match feature weights."
    return 

# reviewed
def _weight_X(X: DF_ARR_TYPE, feat_weights: Optional[SER_ARR_TYPE]) -> np.ndarray:
    """
    Apply feature weights to the input data matrix `X`. If `feat_weights` is `None`, no 
    weights are applied. It is assumed that dimensions match.

    Parameters
    ----------
    X : DataFrame or ndarray
        The input data matrix.

    feat_weights : Series or ndarray, optional
        The array of feature weights. If `None`, no weighting is applied.

    Returns
    -------
    np.ndarray
        The weighted data matrix, with the same shape as `X`.
    """
    X_arr = check_2d_array(X)
    if feat_weights is None: return X_arr
    # Apply feature weights
    feat_weights_arr = check_1d_array(feat_weights)
    return X_arr * feat_weights_arr
        
# reviewed
def check_X_with_feat_weights(X: DF_ARR_TYPE, feat_weights: Optional[SER_ARR_TYPE]) -> np.ndarray:
    """
    Process the input data matrix `X` and feature weights, returning a weighted version of `X`.

    Parameters
    ----------
    X : DataFrame or ndarray
        The input data matrix.

    feat_weights : Series or ndarray, optional
        The array of feature weights. If `None`, no weighting is applied.

    Returns
    -------
    np.ndarray
        The weighted data matrix.
    """
    # Validate that the dimensions of X and feat_weights match
    _valid_shape_X_feat_weights(X, feat_weights)
    # Apply feature weights to X
    return _weight_X(X, feat_weights)

#################################
## model code
#################################

class JumpModel(BaseClusteringAlgo):
    """
    Statistical jump model estimation, supporting both discrete and continuous models.

    This class provides methods for fitting and predicting with jump models, using coordinate 
    descent for optimization. Both discrete and continuous models are supported, with optional 
    feature weighting and state sorting.

    Parameters
    ----------
    n_components : int, default=2
        The number of components (states) in the model.

    jump_penalty : float, default=0.
        Penalty term (`lambda`) applied to state transitions in both discrete and continuous models.

    cont : bool, default=False
        If `True`, the continuous jump model is used. Otherwise, the discrete model is applied.

    grid_size : float, default=0.05
        The grid size for discretizing the probability simplex. Only relevant for the continuous model.

    mode_loss : bool, default=True
        Whether to apply the mode loss penalty. Only relevant for the continuous model.

    random_state : int or RandomState, optional (default=None)
        Random number seed for reproducibility.

    max_iter : int, default=1000
        Maximum number of iterations for the coordinate descent algorithm during model fitting.

    tol : float, default=1e-8
        Stopping tolerance for the improvement in objective value during optimization.

    n_init : int, default=10
        Number of initializations for the model fitting process.

    verbose : int, default=0
        Controls the verbosity of the output. Higher values indicate more verbose output.

    Attributes
    ----------
    centers_ : ndarray of shape (n_c, n_f)
        The cluster centroids estimated during model fitting.

    labels_ : Series or ndarray
        In-sample fitted optimal label sequence.

    proba_ : DataFrame or ndarray
        In-sample fitted optimal probability matrix.

    ret_, vol_ : Series or ndarray
        The average return (`ret_`) and volatility (`vol_`) for each state. These attributes 
        are available only if `ret_ser` is provided to the `.fit()` method.

    transmat_ : ndarray of shape (n_c, n_c)
        The estimated transition probability matrix between states.

    val_ : float
        The optimal value of the loss function.
    """
    # reviewed
    def __init__(self,
                 n_components: int = 2, 
                 jump_penalty: float = 0., 
                 cont: bool = False, 
                 grid_size: float = 0.05, 
                 mode_loss: bool = True, 
                 random_state = RANDOM_STATE, 
                 max_iter: int = 1000, 
                 tol: float = 1e-8, 
                 n_init: int = 10, 
                 verbose: int = 0):
        super().__init__(int(n_components), n_init, max_iter, tol, random_state, verbose)
        self.jump_penalty = jump_penalty
        self.cont = cont
        self.grid_size = grid_size
        self.mode_loss = mode_loss
        self.alpha = 2  # the power raised to the jump penalty in CJM

    # reviewed           
    def check_jump_penalty_mx(self) -> np.ndarray:
        """
        Initialize the jump penalty matrix for state transitions.

        - For the discrete model, the state space is {0, 1, ..., n_c - 1}, and the scalar 
          `jump_penalty` is converted into a matrix.
        - For the continuous model, `jump_penalty` is multiplied by the pairwise L1 distance 
          between probability vectors. Optionally applies a mode loss penalty.

        Returns
        -------
        np.ndarray
            The jump penalty matrix to be used in the model.
        """
        assert is_numbers(self.jump_penalty)
        if not self.cont:
            self.prob_vecs = None      # useful in the E step to tell whether the model is continuous/discrete.
            jump_penalty_mx = jump_penalty_to_mx(self.jump_penalty, self.n_components) 
        else:    # continuous model
            self.prob_vecs = discretize_prob_simplex(self.n_components, self.grid_size)   # state space. useful for computing L mx in E step
            pairwise_l1_dist = cdist(self.prob_vecs, self.prob_vecs, 'cityblock')/2
            jump_penalty_mx = self.jump_penalty * (pairwise_l1_dist ** self.alpha)
            if self.mode_loss:      # adding mode loss ensures that the penalty mx has correspondence with a TPM. i.e. sum(exp(- )) of every row leads to the same value.
                mode_loss = logsumexp(-jump_penalty_mx, axis=1, keepdims=True)
                mode_loss -= mode_loss[0]     # offset a constant
                jump_penalty_mx += mode_loss
        self.jump_penalty_mx = jump_penalty_mx      # to be used in `.predict()`  & `.predict_proba()`
        return jump_penalty_mx
    
    # reviewed
    def check_X_predict_func(self, X: DF_ARR_TYPE) -> np.ndarray:
        """
        Validate the input data `X` for all prediction methods (but not for fitting), 
        and apply feature weighting if applicable. Assumes that the model has already 
        been fitted.

        This method overrides the superclass method.

        Parameters
        ----------
        X : DataFrame or ndarray
            The input data matrix.

        Returns
        -------
        np.ndarray
            The weighted input data matrix, if feature weights are provided.
        """
        self.is_shape_match_X_centers(X)
        feat_weights = getattr_(self, "feat_weights")
        return check_X_with_feat_weights(X, feat_weights)
    
    # reviewed
    def fit(self, 
            X: DF_ARR_TYPE, 
            ret_ser: Optional[SER_ARR_TYPE] = None, 
            feat_weights: Optional[SER_ARR_TYPE] = None,
            sort_by: Optional[str] = "cumret"):
        """
        Fit the jump model using the coordinate descent algorithm.

        The states are sorted by the specified criterion: ["cumret", "vol", "freq", "ret"].
        The Viterbi algorithm is optionally used for state assignment. This choice does 
        not impact the final numerical results but may affect computational speed.

        Parameters
        ----------
        X : DataFrame or ndarray
            The input data matrix.

        ret_ser : Series or ndarray, optional
            A return series used for sorting states and calculating state-specific returns 
            and volatilities.

        feat_weights : Series or ndarray, optional
            Feature weights to apply to the input data matrix.

        sort_by : ["cumret", "vol", "freq", "ret"], optional (default="cumret")
            Criterion for sorting the states.
        """
        # valid feat weights
        valid_feat_weights(feat_weights)
        # check X
        X_arr = check_X_with_feat_weights(X, feat_weights)
        # save valid feat weights
        self.feat_weights = feat_weights
        # get attributes
        n_c = self.n_components
        max_iter = self.max_iter
        tol = self.tol
        verbose = self.verbose
        # make sure the state space, and compute the penalty matrix used for the E step
        jump_penalty_mx = self.check_jump_penalty_mx()
        # init centers
        init_centers_values = self.init_centers(X_arr)
        # the best results over all initializations, compare to it in the last part of each iteration
        best_val = np.inf
        best_res = {}   # store: "centers_", "proba_", "labels_".
        best_res['labels_'] = None # "labels_" is not always 0/1, but the labels of the state space (candidate prob vecs)
                                   #  it is only used to compare whether two inits lead to the same estimation. the final `labels_` is based on `proba_.argmax(axis=1)`.
        # iter over all the initializations
        for n_init_, centers_ in enumerate(init_centers_values):
            # initialize the labels and value in the previous iteration.
            labels_pre, val_pre = None, np.inf
            # do one E step
            proba_, labels_, val_ = do_E_step(X_arr, centers_, jump_penalty_mx, prob_vecs=self.prob_vecs)
            num_iter = 0
            # iterate between M and E steps
            while (num_iter < max_iter and (not is_same_clustering(labels_, labels_pre)) and val_pre - val_ > tol):
                # update
                num_iter += 1
                labels_pre, val_pre = labels_, val_
                # M step: update centers
                centers_ = weighted_mean_cluster(X_arr, proba_) 
                # E step
                proba_, labels_, val_ = do_E_step(X_arr, centers_, jump_penalty_mx, prob_vecs=self.prob_vecs)
            if verbose: print(f"{n_init_}-th init. val: {val_}")
            # compare with previous initializations
            if (not is_same_clustering(best_res['labels_'], labels_)) and val_ < best_val:
                best_idx = n_init_
                best_val = val_
                # save model attributes
                best_res['centers_'] = centers_
                best_res['labels_'] = labels_   # only used to compare with later iters, won't permutate
                best_res['proba_'] = proba_
        self.val_ = best_val
        if verbose: print(f"{best_idx}-th init has the best value: {best_val}.")
        # sort states
        sort_states_from_ret(ret_ser, X, best_res, sort_by=sort_by)
        # save attributes
        if ret_ser is not None:
            self.ret_ = best_res["ret_"]
            self.vol_ = best_res["vol_"]
        self.centers_ = best_res['centers_']        # weighted centers
        self.proba_ = raise_JM_proba_to_df(best_res['proba_'], X)
        self.labels_ = reduce_proba_to_labels(self.proba_)
        self.transmat_ = empirical_trans_mx(self.labels_, n_components=n_c)
        return self
        
    # reviewed
    def predict_proba_online(self, X: DF_ARR_TYPE) -> DF_ARR_TYPE:
        """
        Predict the probability of each state in an online fashion, where the prediction 
        for the i-th row is based only on data prior to that row.

        Parameters
        ----------
        X : DataFrame or ndarray
            The input data matrix.

        Returns
        -------
        DataFrame or ndarray
            The predicted probabilities for each state.
        """
        X_arr = self.check_X_predict_func(X)
        value_mx = do_E_step(X_arr, self.centers_, self.jump_penalty_mx, self.prob_vecs, return_value_mx=True)
        labels_ = value_mx.argmin(axis=1)
        proba_ = raise_JM_labels_to_proba(labels_, self.n_components, self.prob_vecs)
        return raise_JM_proba_to_df(proba_, X)
    
    # reviewed
    def predict_online(self, X: DF_ARR_TYPE) -> SER_ARR_TYPE:
        """
        Predict the state in an online fashion, where the prediction for the i-th row 
        is based only on data prior to that row.

        Parameters
        ----------
        X : DataFrame or ndarray
            The input data matrix.

        Returns
        -------
        Series or ndarray
            The predicted state labels for each sample.
        """
        return reduce_proba_to_labels(self.predict_proba_online(X))
    
    # reviewed
    def predict_proba(self, X: DF_ARR_TYPE) -> DF_ARR_TYPE:
        """
        Predict the probability of each state, using all available data in `X`.

        Parameters
        ----------
        X : DataFrame or ndarray
            The input data matrix.

        use_viterbi : bool, optional (default=True)
            Whether to use the Viterbi solver.

        Returns
        -------
        DataFrame or ndarray
            The predicted probabilities for each state.
        """
        X_arr = self.check_X_predict_func(X)
        proba_, _, _ = do_E_step(X_arr, self.centers_, self.jump_penalty_mx, self.prob_vecs)
        return raise_JM_proba_to_df(proba_, X)

    # reviewed
    def predict(self, X: DF_ARR_TYPE) -> SER_ARR_TYPE:
        """
        Predict the state for each sample, using all available data in `X`.

        Parameters
        ----------
        X : DataFrame or ndarray
            The input data matrix.

        use_viterbi : bool, optional (default=True)
            Whether to use the Viterbi solver.

        Returns
        -------
        Series or ndarray
            The predicted state labels for each sample.
        """
        return reduce_proba_to_labels(self.predict_proba(X))
