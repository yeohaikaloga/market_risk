import pandas as pd
from pandas import Series
from utils.date_utils import get_prev_biz_days_list


class VaRCalculator:

    @staticmethod
    def calculate_historical_var(lookback_df: pd.DataFrame, inverse_df: pd.DataFrame, cob_date: str,
                                 percentile: float = 95, window: int = 260) -> float:
        """
        Calculate historical VaR based on the worse (max) of lookback and inverse PnLs over a window.

        Args:
            date (str): Business date (used only for fallback).
            lookback_df (pd.DataFrame or pd.Series): Lookback PnL data.
            inverse_df (pd.DataFrame or pd.Series): Inverse PnL data.
            percentile (float): Confidence level (e.g., 95 = 5% tail).
            window (int): Lookback window size in days.

        Returns:
            float: Historical VaR (positive number).
        """

        if not (0 < percentile < 100):
            raise ValueError("Percentile must be between 0 and 100 (exclusive).")

        if lookback_df.empty or inverse_df.empty:
            print("Warning: One or both input DataFrames are empty. Returning 0.")
            return 0.0

        lookback_total = lookback_df if isinstance(lookback_df, Series) else lookback_df.sum(axis=1)
        inverse_total = inverse_df if isinstance(inverse_df, Series) else inverse_df.sum(axis=1)

        # Ensure index is datetime
        lookback_total.index = pd.to_datetime(lookback_total.index)
        inverse_total.index = pd.to_datetime(inverse_total.index)

        days_list = get_prev_biz_days_list(cob_date, window + 1)
        days_list_dt = pd.to_datetime(days_list)

        # Filter to matching dates only
        lookback_tail = lookback_total.loc[lookback_total.index.isin(days_list_dt)]
        inverse_tail = inverse_total.loc[inverse_total.index.isin(days_list_dt)]

        # Ensure final filtering returns enough data
        if lookback_tail.empty or inverse_tail.empty:
            print("Warning: No data after filtering to days_list. Returning 0.")
            return 0.0

        tail_q = (100 - percentile) / 100.0
        lookback_var = -lookback_tail.quantile(tail_q)
        inverse_var = -inverse_tail.quantile(tail_q)

        return max(lookback_var, inverse_var)
