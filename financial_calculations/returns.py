import pandas as pd
import numpy as np


def relative_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """
    Calculate relative returns (percentage change) for a Series or DataFrame.
    """
    if isinstance(prices, pd.Series):
        return prices.pct_change(fill_method=None)
    elif isinstance(prices, pd.DataFrame):
        return prices.pct_change(fill_method=None)
    else:
        raise TypeError("Input must be a pandas Series or DataFrame.")


def absolute_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """
    Calculate absolute returns (difference) for a Series or DataFrame.
    """
    if isinstance(prices, pd.Series):
        return prices.diff(fill_method=None)
    elif isinstance(prices, pd.DataFrame):
        return prices.diff(fill_method=None)
    else:
        raise TypeError("Input must be a pandas Series or DataFrame.")


def log_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """
    Calculate log returns for a Series or DataFrame.
    """
    if isinstance(prices, pd.Series):
        return np.log(prices).diff(fill_method=None)
    elif isinstance(prices, pd.DataFrame):
        return np.log(prices).diff(fill_method=None)
    else:
        raise TypeError("Input must be a pandas Series or DataFrame.")
