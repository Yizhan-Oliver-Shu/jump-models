"""
module for sparse jump model estimation
"""

from .utils import *
from .jump import *
from numpy.linalg import norm

############################
## lasso problem for w
############################

# reviewed
def binary_search_decrease(func, left: float, right: float, value: float,  *args, tol=1e-6, max_iter=100, **kwargs):
    """
    binary search for a decreasing function.
    """
    if value >= func(left): return left
    if value <= func(right): return right
    # 
    gap = right-left
    num_iter = 0
    while (gap > tol and num_iter < max_iter):
        # print(f"{left}, {right}")
        num_iter += 1
        middle = (right + left) / 2
        func_call = func(middle, *args, **kwargs)
        if func_call < value:
            right = middle
        elif func_call > value:
            left = middle
        else:
            return middle
        gap /= 2
    if num_iter < max_iter:
        return middle
    raise Exception("None convergence. must be math error.")
    
# reviewed
def soft_thres_l2_normalized(x: SER_ARR_TYPE, thres: float = 0.) -> SER_ARR_TYPE:
    """
    soft thresholding for a (nonneg) vec x. normalize to to have unit length.
    """
    y = np.maximum(0, x-thres)
    y_norm = norm(y)
    assert y_norm > 0
    return y / y_norm

# reviewed
def solve_lasso(a: SER_ARR_TYPE, s: float):
    """
    solve the lasso problem involved in updating the feature weight vector.
    """
    assert s >= 1.
    a_arr = check_1d_array(a)
    left, right = 0., np.unique(a_arr)[-2]  # right is the second largest element of `a`
    if right < 1e-6: thres_sol = 0.
    else:
        func = lambda thres: soft_thres_l2_normalized(a_arr, thres).sum()
        thres_sol = binary_search_decrease(func, left, right, s)
    w = soft_thres_l2_normalized(a_arr, thres_sol)
    return raise_arr_to_pd_obj(w, a)    #w if not is_ser_df(a) else pd.Series(w, index=a.index)

# reviewed
def compute_BCSS(X: DF_ARR_TYPE, proba_: DF_ARR_TYPE, centers_: np.ndarray = None) -> SER_ARR_TYPE:
    """
    compute the between cluster sum of squares.
    """
    X_arr, proba_arr = check_2d_array(X), check_2d_array(proba_)
    if centers_ is None: centers_ = weighted_mean_cluster(X_arr, proba_arr, fill_na_mean=None)
    BCSS = proba_arr.sum(axis=0) @ ((centers_ - X_arr.mean(axis=0))**2)
    valid_no_nan(BCSS)
    return raise_arr_to_pd_obj(BCSS, X, index_key="columns")

############################
## SJM
############################

class SparseJumpModel(BaseEstimator):
    """
    class for SJM implementation
    """
    def __init__(self,
                 n_components: int = 2, 
                 n_feats: float = 100.,
                 jump_penalty: float = 0, 
                 cont: bool = False, 
                 grid_size: float = 0.05, 
                 mode_loss: bool = True, 
                 random_state = None, 
                 max_iter: int = 10, 
                 w_tol: float = 1e-4, 
                 jm_max_iter: int = 1000,
                 jm_tol: float = 1e-8,
                 jm_n_init: int = 10,
                 verbose: bool = False):
        """
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
        self.n_feats = n_feats
        self.jump_penalty = jump_penalty
        self.cont = cont
        self.grid_size = grid_size
        self.mode_loss = mode_loss
        self.random_state = random_state
        self.max_iter = max_iter
        self.w_tol = w_tol
        self.jm_max_iter = jm_max_iter
        self.jm_tol = jm_tol
        self.jm_n_init = jm_n_init
        self.verbose = verbose

    def init_jm(self):
        """
        initialize the JM instance.
        """
        jm = JumpModel(n_components=self.n_components,
                       jump_penalty=self.jump_penalty,
                       cont=self.cont,
                       grid_size=self.grid_size,
                       mode_loss=self.mode_loss,
                       random_state=self.random_state,
                       max_iter=self.jm_max_iter,
                       tol=self.jm_tol,
                       n_init=self.jm_n_init)
        self.jm_ins = jm
        return jm
    
    def fit(self, 
            X: DF_ARR_TYPE, 
            ret_ser: SER_ARR_TYPE = None,
            sort_by: str = "ret",
            refit: bool = False):
        #
        X_arr = check_2d_array(X)
        # get attrs
        max_iter = self.max_iter
        w_tol = self.w_tol
        s = np.sqrt(self.n_feats)
        # 
        n_feats_all = X_arr.shape[1]
        w_old = np.ones(n_feats_all)*2  # invalid weight, just for the 1st iter
        w = np.ones(n_feats_all) * 1/np.sqrt(n_feats_all)   # initial weight
        # jm ins
        jm = self.init_jm()
        # 
        n_iter = 0
        centers_ = None
        while (n_iter < max_iter and norm(w-w_old, 1) / norm(w_old, 1) > w_tol):
            # 
            n_iter += 1
            w_old = w
            # fix w, fit JM
            feat_weights = np.sqrt(w)
            if n_iter > 1: jm.centers_ = centers_ * feat_weights
            jm.fit(X_arr, ret_ser=ret_ser, feat_weights=feat_weights, sort_by=sort_by)
            centers_ = weighted_mean_cluster(X_arr, jm.proba_, )
            # solve new w, compute SS on the original data
            BCSS = compute_BCSS(X_arr, jm.proba_, centers_)
            if np.isclose(BCSS, 0).all(): break     # all in one cluster
            w = solve_lasso(BCSS/1e3, s)
            if self.verbose: print(f"w in iter {n_iter}: ", w)
        # best res
        self.w = w
        if refit:   # retain the feats with non-zero weights.
            feat_weights = np.zeros(len(w))
            feat_weights[w > 0] = 1.
            self.feat_weights = feat_weights
            # refit jm instance
            jm.fit(X_arr, ret_ser=ret_ser, feat_weights=feat_weights, sort_by=sort_by)
        else:
            self.feat_weights = np.sqrt(w)
        self.centers_ = weighted_mean_cluster(X_arr, jm.proba_, )
        self.labels_ = raise_arr_to_pd_obj(jm.labels_, X)
        self.proba_ = raise_arr_to_pd_obj(jm.proba_, X, return_as_ser=False)
        return self
    
    def predict_proba(self, X: DF_ARR_TYPE) -> DF_ARR_TYPE:
        return self.jm_ins.predict_proba(X)

    def predict(self, X: DF_ARR_TYPE) -> SER_ARR_TYPE:
        return self.jm_ins.predict(X)

    def predict_proba_online(self, X: DF_ARR_TYPE) -> DF_ARR_TYPE:
        return self.jm_ins.predict_proba_online(X)
    
    def predict_online(self, X: DF_ARR_TYPE) -> SER_ARR_TYPE:
        return self.jm_ins.predict_online(X)
    