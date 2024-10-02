"""
Helpers for engineering the features to be used in JMs.
"""

from utils_dir import *
include_home_dir()

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from jumpmodels.utils import *

############################################
## Feature Engineering
############################################

# reviewed
def compute_ewm_DD(ret_ser: pd.Series, hl: float) -> pd.Series:
    """
    Compute the exponentially weighted moving downside deviation (DD) for a return series.

    The downside deviation is calculated as the square root of the exponentially 
    weighted second moment of negative returns.

    Parameters
    ----------
    ret_ser : pd.Series
        The input return series.

    hl : float
        The halflife parameter for the exponentially weighted moving average.

    Returns
    -------
    pd.Series
        The exponentially weighted moving downside deviation for the return series.
    """
    ret_ser_neg: pd.Series = np.minimum(ret_ser, 0.)
    sq_mean = ret_ser_neg.pow(2).ewm(halflife=hl).mean()
    return np.sqrt(sq_mean)

# reviewed
def feature_engineer(ret_ser: pd.Series, ver: str = "v0") -> pd.DataFrame:
    """
    Engineer a set of features based on a return series.

    This function customizes the feature set according to the specified version string.

    Parameters
    ----------
    ret_ser : pd.Series
        The input return series for feature engineering.

    ver : str
        The version of feature engineering to apply. Only supports "v0".
    
    Returns
    -------
    pd.DataFrame
        The engineered feature set.
    """
    if ver == "v0":
        feat_dict = {}
        hls = [5, 20, 60]
        for hl in hls:
            # Feature 1: EWM-ret
            feat_dict[f"ret_{hl}"] = ret_ser.ewm(halflife=hl).mean()
            # Feature 2: log(EWM-DD)
            DD = compute_ewm_DD(ret_ser, hl)
            feat_dict[f"DD-log_{hl}"] = np.log(DD)
            # Feature 3: EWM-Sortino-ratio = EWM-ret/EWM-DD 
            feat_dict[f"sortino_{hl}"] = feat_dict[f"ret_{hl}"].div(DD)
        return pd.DataFrame(feat_dict)

    # try out your favorite feature sets
    else:
        raise NotImplementedError()

############################################
## DataLoader Class
############################################

class DataLoader(BaseEstimator):
    """
    Class for loading the feature matrix.

    This class loads raw return data, computes features, and filters the data by date.
    
    Parameters
    ----------
    ticker : str
        The ticker symbol for which to load data. Only supports "NDX".

    ver : str
        The version of the feature set to apply. Only supports "v0".

    Attributes
    ----------
    X : pd.DataFrame
        The feature matrix.
    
    ret_ser : pd.Series
        The return series.
    """
    def __init__(self, ticker: str = "NDX", ver: str = "v0"):
        self.ticker = ticker
        self.ver = ver
    
    # reviewed
    def load(self, start_date: DATE_TYPE = None, end_date: DATE_TYPE = None):
        """
        Load the raw return data, compute features, and filter by date range.

        Parameters
        ----------
        start_date : DATE_TYPE, optional
            The start date for filtering the data. If None, no start filtering is applied.

        end_date : DATE_TYPE, optional
            The end date for filtering the data. If None, no end filtering is applied.

        Returns
        -------
        self
            The DataLoader instance with the feature matrix and return series stored in attributes.
        """
        # load raw data
        curr_dir = get_curr_dir()
        ret_ser_raw = pd.read_pickle(f"{curr_dir}/data/{self.ticker}.pkl").ret.dropna()
        # features
        df_features_all = feature_engineer(ret_ser_raw, self.ver)
        
        # filter date
        X = filter_date_range(df_features_all, start_date, end_date)
        valid_no_nan(X)
        # save attributes
        self.X = X
        self.ret_ser = filter_date_range(ret_ser_raw, start_date, end_date)
        # save more useful attributes if needed
        return self
