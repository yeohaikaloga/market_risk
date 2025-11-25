import pandas as pd
import numpy as np


def relative_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """
    Calculate relative returns (percentage change) for a Series or DataFrame.
    Cleans data by removing zeros, infs, and handling NaNs.
    """
    if not isinstance(prices, (pd.Series, pd.DataFrame)):
        raise TypeError("Input must be a pandas Series or DataFrame.")

    if not isinstance(prices.index, pd.DatetimeIndex):
        try:
            prices = prices.copy()
            prices.index = pd.to_datetime(prices.index)
        except Exception:
            raise TypeError("Index must be datetime-like or convertible to datetime.")
    prices = prices.sort_index()

    # Replace zero prices with NaN to avoid divide-by-zero
    prices = prices.replace(0, np.nan)

    # Calculate percentage change
    returns = prices.pct_change(fill_method=None)

    # Replace inf and -inf with NaN
    returns = returns.replace([np.inf, -np.inf], np.nan)

    return returns


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
