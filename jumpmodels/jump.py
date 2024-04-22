"""
module of JMs and CJMs.
Sparse versions are in module `sparse_jump.py`.
"""

import numpy as np
from itertools import product
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.cluster import kmeans_plusplus
from hmmlearn._hmmc import viterbi

from .utils import *

#################################
## model helpers
#################################

# reviewed
def jump_penalty_to_mx(jump_penalty: float, n_c: int) -> np.ndarray:
    """
    raise a scalar `jump_penalty` into a penalty matrix.
    """
    # assert is_numbers(jump_penalty)
    return jump_penalty * (np.ones((n_c, n_c)) - np.eye(n_c))   # default dtype is float

# reviewed
def discretize_prob_simplex(n_c, grid_size) -> np.ndarray:
    """
    sample all the grid points on a simplex.
    Combinatorial complexity!
    """
    N = int(1/grid_size)
    tuples = filter(lambda x: sum(x)==N, product(range(N+1), repeat = n_c))
    lst = np.array(list(tuples)[::-1], dtype=float)/N
    return lst

# reviewed
def dp_viterbi(loss_mx, penalty_mx):
    r"""
    solve the dp problem involved in E step calculation (hard assignment) via the implementation `_hmmc.viterbi` in `hmmlearn`.
    This function is written in C, and is much faster than our python implementation, but only when # of states is < ~100 (120 is the empirical threshold).
    
    $$\min \sum_{t=0}^{T-1} L(t, s_t) + \sum_{t=1}^{T-1} \Lambda(s_{t-1}, s_t).$$

    Parameters:
    ---------------------------
    loss_mx: array, (n_s, n_c).
        the loss matrix of (l(t, k)).
    penalty_mx: array, (n_c, n_c).
        the jump penalty matrix
    """
    TPM = np.exp(-penalty_mx)
    pi = np.ones(len(penalty_mx))
    neg_value_opt, assign = viterbi(pi, TPM, -loss_mx)
    return assign, -neg_value_opt

# reviewed
def dp(loss_mx: np.ndarray, penalty_mx: np.ndarray, use_viterbi: bool = True, return_value_mx: bool = False):
    r"""
    solve the dp problem involved in E step calculation (hard assignment):
    
    $$\min \sum_{t=0}^{T-1} L(t, s_t) + \sum_{t=1}^{T-1} \Lambda(s_{t-1}, s_t).$$

    This function cannot easily be sped up by using numba. numba has a problem with `.min(axis=0)`.
    
    Parameters:
    ---------------------------
    loss_mx: array, (n_s, n_c).
        the loss matrix of (l(t, k)).
    penalty_mx: array, (n_c, n_c).
        the jump penalty matrix
    """
    n_s, n_c = loss_mx.shape
    assert penalty_mx.shape == (n_c, n_c)
    if (not return_value_mx) and use_viterbi:
        return dp_viterbi(loss_mx, penalty_mx)
    # if return_value_mx==True or use_viterbi == False
    # our own python implementation
    values, assign = np.empty((n_s, n_c)), np.empty(n_s, dtype=np.int32)
    # initial
    values[0] = loss_mx[0]
    # dp 
    for t in range(1, n_s):
        values[t] = loss_mx[t] + (values[t-1][:, np.newaxis] + penalty_mx).min(axis=0) # values[t-1][:, np.newaxis] turns the (t-1)-th row into a column
    if return_value_mx: return values
    # return_value_mx==False and use_viterbi == False
    # find optimal path backwards
    assign[-1] = values[-1].argmin()
    value_opt = values[-1, assign[-1]]
    # traceback
    for t in range(n_s - 1, 0, -1):
        assign[t-1] = (values[t-1] + penalty_mx[:, assign[t]]).argmin()
    return assign, value_opt

# reviewed
def do_E_step(X: np.ndarray, 
              centers_: np.ndarray, 
              penalty_mx: np.ndarray, 
              prob_vecs: np.ndarray = None, 
              return_value_mx: bool = False) -> tuple[np.ndarray, np.ndarray, float]:
    """
    do a single E step. compute the distance mx, and call viterbi solver.
    returns `proba_, labels_, val_`. here `labels_` is that on the corresponding state space.
    """
    n_c = len(centers_) # (n_c, n_f)
    assert (not np.isnan(centers_).any())
    # compute loss matrix
    loss_mx = .5 * cdist(X, centers_, "sqeuclidean")    # (n_s, n_c)
    if prob_vecs is not None:    # cont model, (N, n_c)
        loss_mx = loss_mx @ prob_vecs.T     # each pair of loss between period t and candidate vector, (n_s, N)
    if return_value_mx: return dp(loss_mx, penalty_mx, use_viterbi=False, return_value_mx=True)
    labels_, val_ = dp(loss_mx, penalty_mx)     # output labels_ is of type int, not np.int32
    if prob_vecs is not None: # cont model
        proba_ = prob_vecs[labels_]
    else:    # discrete model
        proba_ = raise_labels_into_proba(labels_, n_c)
    return proba_, labels_, val_    # the returned proba_ must be a valid proba arr

# reviewed
def init_centers(X: np.ndarray, n_c, n_init=10, init="k-means++", random_state=None) -> list[np.ndarray]:
    """
    initialize the centers, by k-means++, for `n_init` times.
    """
    random_state = check_random_state(random_state)
    if init == "k-means++":
        centers = [kmeans_plusplus(X, n_c, random_state=random_state)[0] for _ in range(n_init)]    # since the first sample is uniformly sampled from all points, given large sample size all initializations will be different w.h.p.
        return centers    #np.array(centers)     # (n_init, n_c, n_f)
    else:
        raise NotImplementedError()

#################
## sort 
#################

# reviewed
def sort_param_dict_from_idx(params, idx):
    """
    sort a dict of params from a given idx arr.

    Expected shape of every param:
    - 'ret_': (n_c,)
    - 'vol_': (n_c,)
    - 'means_' / 'centers_': (n_c, n_f)
    - 'transmat_': (n_c, n_c)
    - 'startprob_': (n_c,)
    - 'proba_': (n_s, n_c)
    """
    # ret, vol, means, centers, startprob, only need to permute `axis=0`
    for key in ['ret_', 'vol_', 'means_', 'centers_', 'startprob_']:
        if key in params: params[key] = params[key][idx]
    # transmat, need to permute both `axis=0 & 1`
    if 'transmat_' in params: params['transmat_'] = params['transmat_'][idx][:, idx]
    # proba, need to permute `axis=1`
    if 'proba_' in params: params['proba_'] = params['proba_'][:, idx]
    if "covars_" in params: 
        assert params['covars_'].shape[1] == 1
        params['covars_'] = params['covars_'][idx]
    return params
 
# reviewed
def sort_param_dict(params, by='ret'):
    """
    sort the states by some criterion (e.g. decreasing mean/increasing vol), and permute all the parameters according to the order.
    now only supports sorting by increasing vol (as it is believed in the literature to be the most distinguishable characteristic of equity market cycles). nan vol will be sorted to the end.

    Expected shape of every param:
    - 'ret_': (n_c,)
    - 'vol_': (n_c,)
    - 'means_' / 'centers_': (n_c, n_f)
    - 'covars_' / 'scales': (n_c, ...). the shape of `covars_` depends on the cov type, where for cov type with "tied", `covars_` don't need to be permuted. this function will not check this. please take care of the inputs
    - 'transmat_': (n_c, n_c)
    - 'startprob_': (n_c,)
    - 'proba_': (n_s, n_c)
    """
    assert by in ["ret", "vol"]
    if by == 'vol':
        assert 'vol_' in params
        idx = np.argsort(params['vol_'])   # vol in increasing order. may contain nans
                                           # `np.argsort` accepts np.nan in inputs, and puts them to the last few elements, i.e. missing clusters will be put to the end.
        params = sort_param_dict_from_idx(params, idx)
    elif by == "ret":
        assert "ret_" in params
        idx = np.argsort(params['ret_'])[::-1]
        return sort_param_dict_from_idx(params, idx)
    else:
        raise NotImplementedError("only supports sorting by 1. increasing vol, 2. decreasing ret.") 
    return params

#################################
## model code
#################################

class JumpModel(BaseEstimator):
    """
    Statistical jump model estimation. Includes both the discrete and continuous model implementation.
    """
    def __init__(self,
                 n_components: int = 2, 
                 jump_penalty: float = 0, 
                 cont: bool = False, 
                 grid_size: float = 0.05, 
                 mode_loss: bool = True, 
                 random_state = None, 
                 max_iter: int = 1000, 
                 tol: float = 1e-8, 
                 n_init: int = 10, 
                 verbose: bool = False):
        """
        Run several initializations of centers from k-means++, and return the best one.
        
        Parameters:
        ----------------------------------
        n_components: int, default = 2
            number of components.
        cont: boolean
            if True run the continuous jump model. otherwise the discrete model.
        jump_penalty: float, default = 0.
            `jump_penalty` is the lambda for both the discrete and continous model.
        grid_size: float
            grid size to discretize the probability simplex. only useful for the continuous model.
        mode_loss: boolean
            whether to add the mode loss penalty. only useful for the continuous model.
        covariance_type: [None, "tied_diag", "diag"]
            The covaraiance mx type:
            - None: no cov mx structure, squared l2 norm on the raw data is used as the loss function.
            - "tied_diag": the same diagonal cov mx is imposed on every cluster. Equivalent to first standardize the data matrix, and use squared l2 norm as the loss function.
            - "diag": learn a diagonal cov mx for every cluster. equivalent to computing the feat std within each cluster, and use it to standardize the data when computing the loss function for the corresponding cluster.
        random_state:

        max_iter: int, default = 1000
            maximal number of iteration in each EM algo.    
        tol: float, default=1e-8.
            tolerance for the improvement of objective value    
        n_init: int, default = 10
            number of different initializations.
        verbose: boolean

        """
        self.n_components = int(n_components)
        self.jump_penalty = jump_penalty
        self.cont = cont
        self.grid_size = grid_size
        self.mode_loss = mode_loss
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.verbose = verbose
        self.alpha = 2  # the power raised to the jump penalty in CJM
                
    def check_jump_penalty_mx(self) -> np.ndarray:
        """
        make sure the state space of the model, and compute the penalty matrix on the state space. return the panelty matrix.
        - if discrete model, state spce is {0, 1, ..., n_c-1}. raise the float `jump_penalty` to a matrix.
        - if cont model, multiply `jump_penalty` with alpha power of half pairwiase l1-dist. considers mode loss.
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
    
    def init_centers(self, X: np.ndarray) -> np.ndarray:
        centers = init_centers(X, self.n_components, self.n_init, random_state=self.random_state)
        if hasattr(self, "centers_"):
            centers_old = self.centers_
            if is_no_nan(centers_old) and centers_old.shape == (self.n_components, X.shape[1]): centers.append(centers_old)  # use previously fitted value as one initial center value
        return np.array(centers)
    
    def fit(self, 
            X: DF_ARR_TYPE, 
            ret_ser: SER_ARR_TYPE = None, 
            feat_weights: SER_ARR_TYPE = None,
            sort_by: str = "ret"):
        """
        fit the jump model by the EM-type algo.
        `ret_ser` is only used to calculate the expected ret & vol for each cluster. if not provided the algo will sort by decreasing freq.
        """
        # X mx
        X_arr = check_2d_array(X)
        # feat weights
        if feat_weights is not None:
            feat_weights = check_1d_array(feat_weights)
            assert (feat_weights >= 0.).all()
            assert len(feat_weights) == X_arr.shape[1]
            self.feat_weights = feat_weights
            X_arr *= feat_weights
        else:
            self.feat_weights = None
        # get attributes
        n_c = self.n_components
        max_iter = self.max_iter
        tol = self.tol
        verbose = self.verbose
        # make sure the state space, and compute the penalty matrix used for the E step
        jump_penalty_mx = self.check_jump_penalty_mx()
        # init centers
        init_values = self.init_centers(X_arr)
        # the best results over all initializations, compare to it in the last part of each iteration
        best_val = np.inf
        best_res = {}   # store: "centers_", "proba_", "labels_".
        best_res['labels_'] = None # "labels_" is not always 0/1, but the labels of the state space (candidate prob vecs)
                                   #  it is only used to compare whether two inits lead to the same estimation. the final `labels_` is based on `proba_.argmax(axis=1)`.
        # iter over all the initializations
        for n_init_, centers_ in enumerate(init_values):      #range(n_init):
            # labels and value in the previous iteration.
            labels_pre, val_pre = None, np.inf
            # do one E step
            proba_, labels_, val_ = do_E_step(X_arr, centers_, jump_penalty_mx, prob_vecs=self.prob_vecs)          
            num_iter = 0
            # iterate between M and E steps
            while (num_iter < max_iter and not is_same_clustering(labels_, labels_pre) and val_pre - val_ > tol):
                # update
                num_iter += 1
                labels_pre, val_pre = labels_, val_
                # M step: update centers
                centers_ = weighted_mean_cluster(X_arr, proba_, fill_na_mean=centers_) 
                # E step
                proba_, labels_, val_ = do_E_step(X_arr, centers_, jump_penalty_mx, prob_vecs=self.prob_vecs)
            if verbose: print(f"{n_init_}-th init. val: {val_}")
            # compare with previous initializations
            if not is_same_clustering(best_res['labels_'], labels_) and val_ < best_val:
                best_idx = n_init_
                best_val = val_
                # save model attributes
                best_res['centers_'] = centers_
                best_res['labels_'] = labels_   # only used to compare with later iters, won't permutates
                best_res['proba_'] = proba_
        self.val_ = best_val
        if verbose: print(f"{best_idx}-th init has the best value: {best_val}.")
        # explicitly put NA values if there is any
        best_res['centers_'] = weighted_mean_cluster(X_arr, best_res['proba_'], fill_na_mean=None)
        # sort states
        if ret_ser is not None: 
            # valid inputs
            if is_ser_df(ret_ser) and is_ser_df(X): ret_ser_arr = check_1d_array(align_index(ret_ser, X))
            else: 
                assert len(X) == len(ret_ser)
                ret_ser_arr = check_1d_array(ret_ser)
            # compute mean & vol for each cluster
            best_res['ret_'], best_res['vol_'] = weighted_mean_std_cluster(ret_ser_arr, best_res['proba_'])
            # the best parameters sorted by vol
            best_res = sort_param_dict(best_res, by=sort_by)
            # save attributes
            self.ret_ = best_res["ret_"]
            self.vol_ = best_res["vol_"]
        else:
            freq = best_res['proba_'].sum(axis=0)
            best_res = sort_param_dict_from_idx(best_res, np.argsort(freq)[::-1])
        # save results
        self.centers_ = best_res['centers_']
        self.proba_ = raise_arr_to_pd_obj(best_res['proba_'], X, return_as_ser=False)
        self.labels_ = reduce_proba_to_labels(self.proba_)
        self.transmat_ = empirical_trans_mx(self.labels_, n_components=n_c)
        return self
    
    def check_input_X(self, X: DF_ARR_TYPE) -> np.ndarray:
        """
        valid input `X` for all `predict` methods (not for `fit` method), and weight features is needed.
        model is assumed to have been fitted.
        """
        X_arr = check_2d_array(X)
        assert X_arr.shape[1] == self.centers_.shape[1]
        if self.feat_weights is not None: X_arr *= self.feat_weights
        return X_arr

    def process_centers_na(self) -> int:
        """
        
        """
        indic = np.isnan(self.centers_)
        if not indic.any(): return self.centers_
        # replace na with inf
        centers_ = self.centers_.copy()
        centers_[indic] = np.inf
        return centers_
        
    def predict_proba(self, X: DF_ARR_TYPE) -> DF_ARR_TYPE:
        X_arr = self.check_input_X(X)
        centers_ = self.process_centers_na()
        proba_, _, _ = do_E_step(X_arr, centers_, self.jump_penalty_mx, self.prob_vecs)
        return raise_arr_to_pd_obj(proba_, X, return_as_ser=False)

    def predict(self, X: DF_ARR_TYPE) -> SER_ARR_TYPE:
        return reduce_proba_to_labels(self.predict_proba(X))

    def predict_proba_online(self, X: DF_ARR_TYPE) -> DF_ARR_TYPE:
        X_arr = self.check_input_X(X)
        centers_ = self.process_centers_na()
        value_mx = do_E_step(X_arr, centers_, self.jump_penalty_mx, self.prob_vecs, return_value_mx=True)
        labels_ = value_mx.argmin(axis=1)
        proba_ = self.prob_vecs[labels_] if self.cont else raise_labels_into_proba(labels_, self.n_components)
        return raise_arr_to_pd_obj(proba_, X, return_as_ser=False)
    
    def predict_online(self, X: DF_ARR_TYPE) -> SER_ARR_TYPE:
        return reduce_proba_to_labels(self.predict_proba_online(X))
    