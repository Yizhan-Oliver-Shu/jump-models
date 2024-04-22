"""
feature engineering for jump model estimation.
"""

import pandas as pd

def load_example_features(ret_ser: pd.Series) -> pd.DataFrame:
    """
    load the features shown in the notebook of example use.
    """
    feat_dict = {}
    hls = [5, 15, 45]
    # vol
    for hl in hls:
        feat_dict[f"vol_{hl}"] = ret_ser.ewm(halflife=hl).std()
    # ret
    for hl in hls:
        feat_dict[f"ret_{hl}"] = ret_ser.ewm(halflife=hl).mean()
    return pd.DataFrame(feat_dict)
