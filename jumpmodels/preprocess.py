"""
Module for data preprocessing.

This module contains classes for scaling and clipping data, with a focus on 
handling pandas DataFrame input/output.

Depends on
----------
utils/ : Modules
"""

from .utils import *

from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

############################################
## Scaler
############################################

# reviewed
class StandardScalerPD(BaseEstimator):
    """
    Provides support for pandas DataFrame input/output with the `StandardScaler()` class.
    
    This class extends the functionality of the standard `StandardScaler` by ensuring that
    the input and output are handled as pandas DataFrames, preserving index and column labels.
    """
    def init_scaler(self):
        """
        Initialize and return the standard `StandardScaler` instance.
        """
        return StandardScaler()
    
    def fit_transform(self, X: DF_ARR_TYPE) -> DF_ARR_TYPE:
        """
        Fit the scaler to the DataFrame and transform it in one step.
        
        Parameters
        ----------
        X : DataFrame or ndarray
            The input DataFrame to be scaled.
            
        Returns
        -------
        DataFrame or ndarray
            The scaled DataFrame.
        """
        return self.fit(X).transform(X)
    
    def fit(self, X: DF_ARR_TYPE):
        """
        Fit the scaler to the input DataFrame.
        
        Parameters
        ----------
        X : DataFrame or ndarray
            The input DataFrame to be used for fitting.
        
        Returns
        -------
        self
        """
        self.scaler = self.init_scaler().fit(X)
        return self

    def transform(self, X: DF_ARR_TYPE) -> DF_ARR_TYPE:
        """
        Transform the input DataFrame using the fitted scaler.
        
        Parameters
        ----------
        X : DataFrame or ndarray
            The input DataFrame to be transformed.
        
        Returns
        -------
        DataFrame or ndarray
            The transformed (scaled) DataFrame.
        """
        return raise_arr_to_pd_obj(self.scaler.transform(X), X, return_as_ser=False)

############################################
## Clipper
############################################

# reviewed
class BaseDataClipper(BaseEstimator):
    """
    Base class for data clippers. 

    This class implements the `.transform()` and `.fit_transform()` methods, but leaves the `.fit()` 
    method to be implemented in subclasses. It is designed to clip data values within a specified range.
    
    Should be inherited by other classes that define the clipping bounds.
    """
    def __init__(self) -> None:
        self.lb = None  # Lower bound, initialized as None. Must be a numpy array.
        self.ub = None  # Upper bound, initialized as None. Must be a numpy array.

    def fit(self, X: DF_ARR_TYPE):
        raise NotImplementedError()

    def fit_transform(self, X: DF_ARR_TYPE) -> DF_ARR_TYPE:
        """
        Fit the clipper and transform the input data in one step.
        
        Parameters
        ----------
        X : DataFrame or ndarray
            The input data to be clipped.

        Returns
        -------
        DataFrame or ndarray
            The clipped data.
        """
        return self.fit(X).transform(X)
    
    def transform(self, X: DF_ARR_TYPE) -> DF_ARR_TYPE:
        """
        Clip the input data using the fitted lower (`lb`) and upper (`ub`) bounds.
        
        Parameters
        ----------
        X : DataFrame or ndarray
            The input data to be clipped.

        Returns
        -------
        DataFrame or ndarray
            The clipped data.
        """
        if self.ub is None and self.lb is None: return X
        return np.clip(X, self.lb, self.ub)

# reviewed
class DataClipperStd(BaseDataClipper):
    """
    Data clipper based on feature standard deviation.

    This class performs winsorization of the data, clipping it within a specified multiple of the 
    feature's standard deviation. The clipping bounds are defined as:
    
    lower bound = mean - (mul * std)
    upper bound = mean + (mul * std)

    Parameters
    ----------
    mul : float, default=3.
        The multiple of the feature's standard deviation used for clipping.

    Attributes
    ----------
    lb : ndarray
        The lower bound for each feature, calculated as mean - (mul * std).
    
    ub : ndarray
        The upper bound for each feature, calculated as mean + (mul * std).
    """
    def __init__(self, mul: float = 3.) -> None:
        super().__init__()
        self.mul = mul

    def fit(self, X: DF_ARR_TYPE):
        """
        Fit the clipper to the data by calculating the clipping bounds based on 
        the mean and standard deviation of each feature.

        Parameters
        ----------
        X : DataFrame or ndarray
            The input data to fit the clipper.

        Returns
        -------
        DataClipperStd
            The fitted clipper instance.
        """
        mul = self.mul
        assert mul > 0, "The multiplier `mul` must be positive."

        mean, std = X.mean(axis=0), X.std(axis=0, ddof=0)
        if is_df(X):
            mean = mean.to_numpy()
            std = std.to_numpy()
        self.lb = mean - mul * std; assert isinstance(self.lb, np.ndarray)
        self.ub = mean + mul * std; assert isinstance(self.ub, np.ndarray)
        return self
