import pandas as pd
from utils.date_utils import no_of_days_list

def calculate_var(date: str, lookback_df: pd.DataFrame, inverse_df: pd.DataFrame, percentile: float = 95,
                  window: int = 260) -> pd.Series:
    """
        Calculate rolling historical Value at Risk (VaR) at any percentile level.

        Args:
            date (str): Business date.
            lookback_df (pd.DataFrame): Main returns data.
            inverse_df (pd.DataFrame): Inverse returns data.
            percentile (float): Confidence level (e.g., 95 for 95% VaR).
            window (int): Rolling window size.

        Returns:
            pd.Series: VaR time series.
    """

    if not (0 < percentile < 100):
        raise ValueError("Percentile must be between 0 and 100 (exclusive).")

    if lookback_df.empty or inverse_df.empty:
        print("Warning: One or both input DataFrames are empty. Returning zeros.")
        index = no_of_days_list(date, window)
        return pd.Series(0.0, index=index, name=f"VaR_{percentile:.1f}")

    lookback_df = lookback_df.sort_index()
    inverse_df = inverse_df.sort_index()

    if isinstance(lookback_df, pd.Series):
        lookback_subtotal = lookback_df
    else:
        lookback_subtotal = lookback_df.sum(axis=1)

    if isinstance(inverse_df, pd.Series):
        inverse_subtotal = inverse_df
    else:
        inverse_subtotal = inverse_df.sum(axis=1)

    # Calculate rolling quantile for tail risk
    tail_quantile = (100 - percentile) / 100.0
    lookback_var = lookback_subtotal.rolling(window).quantile(tail_quantile).abs()
    inverse_var = inverse_subtotal.rolling(window).quantile(tail_quantile).abs()

    # Final VaR is the worse (max) of the two
    var_series = pd.concat([lookback_var, inverse_var], axis=1).max(axis=1)
    var_series.name = f"VaR_{percentile:.1f}"

    return var_series