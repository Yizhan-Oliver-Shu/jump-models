import sys, os
sys.path.append(os.getcwd())

import yfinance as yf
import pandas as pd

def get_data():
    # Define the ticker symbol for NASDAQ Composite
    ticker_symbol = '^IXIC'
    # Fetch NASDAQ Composite data
    prc = yf.download(ticker_symbol)['Close']
    ret = prc.pct_change()
    df = pd.DataFrame({"prc": prc, "ret": ret}, index=prc.index.date).dropna()
    df.index.name = "date"
    # save
    pd.to_pickle(df, "notebooks/nasdaq_example/data.pkl")
    df.to_csv("notebooks/nasdaq_example/data.csv")
    return 

if __name__ == "__main__":
    get_data()