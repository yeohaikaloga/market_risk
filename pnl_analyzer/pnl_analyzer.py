import pandas as pd
import numpy as np


class PnLAnalyzer:
    """
    Analyzes PnL data, merging it with position data for filtering and pivoting.
    """

    def __init__(self, pnl_df: pd.DataFrame, position_df: pd.DataFrame = None):
        self.pnl_df = pnl_df.copy()
        self.position_df = position_df.copy() if position_df is not None else None
        self._pos_and_pnl_df = self._merge_pos_and_pnl() if position_df is not None else pnl_df

    def __bool__(self):
        """
        Returns True if the underlying data frame is not empty.
        """
        return not self._pos_and_pnl_df.empty
    def _merge_pos_and_pnl(self):
        """
        Merges PnL data with position data on 'position_index'.
        """
        #print("pnl_df.columns:", self.pnl_df.columns)
        #print("position_df.columns:", self.position_df.columns)
        return self.pnl_df.merge(self.position_df, on='position_index', how='left')

    def filter(self, **kwargs):
        """
        Filters the internal DataFrame based on keyword arguments.
        Supports single values, lists/sets, or callable functions for filtering.
        Returns a new PnLAnalyzer object with the filtered PnL data.
        """
        df = self._pos_and_pnl_df.copy()
        for key, value in kwargs.items():
            if callable(value):
                df = df[df[key].apply(value)]
            elif isinstance(value, (list, set, tuple, pd.Series, np.ndarray)):
                if len(value) == 0:
                    continue  # Skip filter if list-like is empty
                df = df[df[key].isin(value)]
            else:
                df = df[df[key] == value]
        common_cols = [col for col in self.pnl_df.columns if col in df.columns]
        filtered_pnl_df = df[common_cols].copy()

        return PnLAnalyzer(filtered_pnl_df, self.position_df)

    def get_position_index(self) -> list:
        """
        Returns a list of unique, non-NaN position indices.
        """
        if 'position_index' not in self._pos_and_pnl_df.columns:
            raise KeyError("Column 'position_index' not found in data.")
        return self._pos_and_pnl_df['position_index'].dropna().unique().tolist()

    def pivot(self, index='region', columns='pnl_date', values='lookback_pnl'):
        df = self._pos_and_pnl_df
        """
        Performs a pivot operation on the underlying data using 'sum' as the aggregation function.
        Supports single strings or lists/tuples of strings for index, columns, and values.
        """
        # Flatten index, columns, and values to lists for uniform checking
        index_cols = index if isinstance(index, (list, tuple)) else [index]
        column_cols = columns if isinstance(columns, (list, tuple)) else [columns]
        value_cols = values if isinstance(values, (list, tuple)) else [values]

        # Check for missing columns
        all_required_cols = index_cols + column_cols + value_cols
        missing_keys = [key for key in all_required_cols if key not in df.columns]
        if missing_keys:
            print("Available columns:", df.columns.tolist())
            raise KeyError(f"Missing pivot key(s): {missing_keys}")

        # Perform pivot
        return df.pivot_table(index=index, columns=columns, values=values, aggfunc='sum')

    def to_excel(self, path):
        # Example: Export grouped pivot or raw data
        with pd.ExcelWriter(path) as writer:
            self._pos_and_pnl_df.to_excel(writer, sheet_name='PnL')
            self.pivot().to_excel(writer, sheet_name='Pivot')
