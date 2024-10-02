"""
This script retrieves the daily closing price data for 
the Nasdaq-100 index from Yahoo Finance via its Python API.

Users do not need to run this script manually, as the return data 
is already saved in `example/Nasdaq/data/`. 
"""

from utils_dir import *

include_home_dir()

import numpy as np
import pandas as pd
import yfinance as yf

from jumpmodels.utils import check_dir_exist

TICKER = "NDX"   # Nasdaq-100 Index

def get_data():
    # download closing prices
    close: pd.Series = yf.download("^"+TICKER)['Close']
    # convert to ret
    ret = close.pct_change()
    # concat as df
    df = pd.DataFrame({"close": close, "ret": ret}, index=close.index.date)
    df.index.name = "date"

    # save
    curr_dir = get_curr_dir()
    data_dir = f"{curr_dir}/data/"; check_dir_exist(data_dir)
    pd.to_pickle(df, f"{data_dir}{TICKER}.pkl")
    np.round(df, 6).to_csv(f"{data_dir}{TICKER}.csv")
    print("Successfully downloaded data for ticker:", TICKER)
    return 

if __name__ == "__main__":
    get_data()