import pandas as pd
from pandas import Series
from utils.date_utils import get_prev_biz_days_list


class VaRCalculator:

    @staticmethod
    def calculate_historical_var(lookback_df: pd.DataFrame, inverse_df: pd.DataFrame, cob_date: str,
                                 percentile: float = 95, window: int = 260) -> float:
        """
        Calculate historical VaR based on the worse (max) of lookback and inverse PnLs over a window.
        This function requires the input PnL DataFrames to be indexed by DATE.

        Args:
            cob_date (str): Business date (used only for fallback).
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

        returns_days_list = get_prev_biz_days_list(cob_date, window)
        returns_days_list_dt = pd.to_datetime(returns_days_list)

        # Filter to matching dates only
        lookback_tail = lookback_total.loc[lookback_total.index.isin(returns_days_list_dt)]
        inverse_tail = inverse_total.loc[inverse_total.index.isin(returns_days_list_dt)]

        # Ensure final filtering returns enough data
        if lookback_tail.empty or inverse_tail.empty:
            print("Warning: No data after filtering to days_list. Returning 0.")
            return 0.0

        tail_q = (100 - percentile) / 100.0
        lookback_var = -lookback_tail.quantile(tail_q)
        inverse_var = -inverse_tail.quantile(tail_q)

        return max(lookback_var, inverse_var)

    @staticmethod
    def calculate_var(lookback_df: pd.DataFrame, inverse_df: pd.DataFrame, simulation_method: str,
                      calculation_method: str, cob_date: str, percentile: float = 95, window: int = 260) -> float:
        """
        Calculates VaR using a single interface, supporting both Historical (hist_sim) and Monte Carlo (mc_sim)
        simulations. The final VaR is derived from the maximum loss across two PnL distributions
        (standard/lookback and inverse/stressed).

        Args:
            lookback_df (pd.DataFrame or pd.Series): Lookback PnL data.
            inverse_df (pd.DataFrame or pd.Series): Inverse PnL data.
            simulation_method (str): 'hist_sim' or 'mc_sim'.
            calculation_method (str): 'linear', 'sensitivity_matrix', or 'repricing'.
                                      (This flag controls upstream PnL generation, but is tracked here.)
            percentile (float): Confidence level (e.g., 95 = 5% tail).
            cob_date (str, optional): Close of Business Date, required for 'hist_sim' for date filtering.
            window (int): Lookback window size in days, required for 'hist_sim' for date filtering.

        Returns:
            float: VaR (positive number, max of the two VaR results).
        """

        simulation_method = simulation_method.lower()
        calculation_method = calculation_method.lower()

        # 1. Input Validation
        if not (0 < percentile < 100):
            raise ValueError("Percentile must be between 0 and 100 (exclusive).")

        if lookback_df.empty or inverse_df.empty:
            print("Warning: One or both input PnL DataFrames are empty. Returning 0.")
            return 0.0

        if simulation_method not in ['hist_sim', 'mc_sim']:
            raise ValueError(f"Unknown simulation_method: {simulation_method}. Must be 'hist_sim' or 'mc_sim'.")

        if simulation_method == 'hist_sim' and (cob_date is None or window is None):
            raise ValueError("For 'hist_sim', both 'cob_date' and 'window' must be provided.")

        if calculation_method not in ['linear', 'sensitivity_matrix', 'repricing', 'taylor_series']:
            print(f"Warning: Calculation method '{calculation_method}' is unknown. Proceeding with VaR calculation.")

        lookback_total = lookback_df if isinstance(lookback_df, Series) else lookback_df.sum(axis=1)
        inverse_total = inverse_df if isinstance(inverse_df, Series) else inverse_df.sum(axis=1)

        if simulation_method == 'hist_sim':
            lookback_total.index = pd.to_datetime(lookback_total.index)
            inverse_total.index = pd.to_datetime(inverse_total.index)

            returns_days_list = get_prev_biz_days_list(cob_date, window)
            returns_days_list_dt = pd.to_datetime(returns_days_list)

            # Filter to matching dates only
            lookback_tail = lookback_total.loc[lookback_total.index.isin(returns_days_list_dt)]
            inverse_tail = inverse_total.loc[inverse_total.index.isin(returns_days_list_dt)]

            # Ensure final filtering returns enough data
            if lookback_tail.empty or inverse_tail.empty:
                print("Warning: No data after filtering to days_list. Returning 0.")
                return 0.0

        elif simulation_method == 'mc_sim':
            lookback_tail = lookback_total
            inverse_tail = inverse_total

        tail_q = (100 - percentile) / 100.0
        lookback_var = -lookback_tail.quantile(tail_q)
        inverse_var = -inverse_tail.quantile(tail_q)

        # VaR is the maximum of the two loss scenarios
        return max(lookback_var, inverse_var)

