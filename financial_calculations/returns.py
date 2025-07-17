import pandas as pd

def relative_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change(fill_method=None)

def absolute_returns(prices: pd.Series) -> pd.Series:
    return prices.diff(fill_method=None)

def log_returns(prices: pd.Series) -> pd.Series:
    import numpy as np
    return np.log(prices).diff(fill_method=None)