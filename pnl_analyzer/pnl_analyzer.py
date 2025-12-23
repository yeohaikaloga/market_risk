import pandas as pd
import numpy as np
from typing import List, Any, Callable, Optional, Union


class PnLAnalyzer:
    """
    Analyzes PnL data, merging it with position data for filtering and pivoting.
    """

    def __init__(self, pnl_df: pd.DataFrame, position_df: pd.DataFrame = None):
        self.pnl_df = pnl_df
        self.position_df = position_df if position_df is not None else None
        if 'position_index' in self.position_df.columns:
            self.position_index = self.position_df['position_index'].dropna().unique().tolist() if position_df is not None else None


    def _get_filtered_dfs(self, position_index_list: List[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Internal helper to get the highly reduced PnL and Position DataFrames.
        """
        # Filter 1: Reduce the metadata table (small)
        filtered_position_df = self.position_df[
            self.position_df['position_index'].isin(position_index_list)
        ].copy()

        # Filter 2: Reduce the massive PnL table
        filtered_pnl_df = self.pnl_df[
            self.pnl_df['position_index'].isin(position_index_list)
        ].copy()

        return filtered_pnl_df, filtered_position_df

    def filter_position_metadata(self, **kwargs: Union[Any, List[Any], Callable]) -> Optional['PnLAnalyzer']:
        """
        Filters the internal DataFrame based on keyword arguments.
        Supports single values, lists/sets, or callable functions for filtering.
        Returns a new PnLAnalyzer object with the filtered PnL data.
        """
        filtered_position_df = self.position_df
        print(f"Applying metadata filters: {kwargs}...")

        for key, value in kwargs.items():
            if key not in filtered_position_df.columns:
                print(f"Warning: Column '{key}' not found in position metadata. Skipping filter.")
                continue

            if len(filtered_position_df) == 0:
                pass
            elif callable(value):
                # Filter using a custom function (e.g., lambda x: x > 100)
                filtered_position_df = filtered_position_df[filtered_position_df[key].apply(value)]
            elif isinstance(value, (list, set, tuple, pd.Series, np.ndarray)):
                # Filter using a list of accepted values
                if len(value) == 0:
                    continue
                filtered_position_df = filtered_position_df[filtered_position_df[key].isin(value)]
            else:
                # Filter using a single exact value
                filtered_position_df = filtered_position_df[filtered_position_df[key] == value]

        filtered_position_index_list = filtered_position_df['position_index'].unique().tolist()

        filtered_pnl_df, filtered_position_df = self._get_filtered_dfs(filtered_position_index_list)
        filtered_analyzer = PnLAnalyzer(position_df=filtered_position_df,
                                        pnl_df=filtered_pnl_df)
        return filtered_analyzer

    def get_position_index(self):
        # If the DataFrame is empty and column-less, or the column is missing
        if 'position_index' not in self.position_df.columns:
            return []

        # If the column exists, return the unique list (which will be empty if the DF is empty)
        return self.position_df['position_index'].unique().tolist()

    def _get_merged_df(self, pnl_df: pd.DataFrame, position_df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs the memory-intensive merge ONLY on the highly filtered DataFrames.
        This is the "Merge Later" step.
        """
        if pnl_df.empty or position_df.empty:
            return pd.DataFrame()

        print(f"Merging reduced datasets: PnL size={len(pnl_df):,} rows, Position size={len(position_df):,} rows.")

        # Perform the merge on the reduced data
        merged_df = pnl_df.merge(
            position_df,
            on='position_index',
            how='inner',
            suffixes=('_pnl', '_pos')
        )
        return merged_df

    def pivot(self, index: Union[str, List[str]] = 'region',
                        columns: Optional[Union[str, List[str]]] = None,
                        values: Union[str, List[str]] = 'lookback_pnl',
                        aggfunc: str = 'sum') -> pd.DataFrame:
        """
        Performs a pivot operation on the underlying data using 'sum' as the aggregation function.
        Supports single strings or lists/tuples of strings for index, columns, and values.
        """

        # 1. Get the highly filtered datasets based on current state
        filtered_pnl_df, filtered_position_df = self._get_filtered_dfs(
            self.position_index
        )

        # 2. Merge them (the "Merge Later" step)
        merged_df = self._get_merged_df(filtered_pnl_df, filtered_position_df)

        if merged_df.empty:
            print("No data left after filtering/merging. Returning empty DataFrame.")
            return pd.DataFrame()

        # 3. Validation and Pivot (Carrying over logic from old pivot)

        # Flatten index, columns, and values to lists for uniform checking
        index_cols = index if isinstance(index, (list, tuple)) else [index]
        column_cols = columns if isinstance(columns, (list, tuple)) else [columns] if columns is not None else []
        value_cols = values if isinstance(values, (list, tuple)) else [values]

        # Check for missing columns
        all_required_cols = index_cols + column_cols + value_cols
        missing_keys = [key for key in all_required_cols if key not in merged_df.columns]

        if missing_keys:
            print("Available columns after merge:", merged_df.columns.tolist())
            raise KeyError(f"Missing pivot key(s) in the merged data: {missing_keys}")

        # Perform pivot
        print(f"Calculating pivot using index={index}, values={values}, aggfunc='{aggfunc}'...")
        pivot_table = merged_df.pivot_table(
            index=index,
            columns=columns,
            values=values,
            aggfunc=aggfunc
        )

        return pivot_table

    def to_excel(self, path):
        # Example: Export grouped pivot or raw data
        with pd.ExcelWriter(path) as writer:
            self.pivot().to_excel(writer, sheet_name='Pivot')
