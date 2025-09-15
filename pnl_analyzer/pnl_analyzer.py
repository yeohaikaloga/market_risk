import pandas as pd
import numpy as np


class PnLAnalyzer:
    def __init__(self, pnl_df: pd.DataFrame, position_df: pd.DataFrame = None):
        self.pnl_df = pnl_df.copy()
        self.position_df = position_df.copy() if position_df is not None else None

        if self.position_df is not None:
            if 'position_index' in self.pnl_df.columns and 'position_index' in self.position_df.columns:
                self._pos_and_pnl_df = self._merge_pos_and_pnl()
            else:
                print("[WARN] Skipping merge: 'position_index' missing in one or both DataFrames.")
                self._pos_and_pnl_df = self.pnl_df.copy()
        else:
            self._pos_and_pnl_df = self.pnl_df.copy()

    def _merge_pos_and_pnl(self):
        #print("pnl_df duplicates:", self.pnl_df.duplicated().sum())
        #print("position_df duplicates:", self.position_df['position_index'].duplicated().sum())
        return self.pnl_df.merge(self.position_df, on='position_index', how='left')

    def filter(self, **kwargs):
        df = self._pos_and_pnl_df.copy()

        for key, value in kwargs.items():
            if callable(value):
                df = df[df[key].apply(value)]
            elif isinstance(value, (list, set, tuple, pd.Series, np.ndarray)):
                if len(value) == 0:
                    continue
                df = df[df[key].isin(value)]
            else:
                df = df[df[key] == value]

        if df.empty:
            print("[INFO] Filter returned empty DataFrame. Returning None.")
            return None

        common_cols = [col for col in self.pnl_df.columns if col in df.columns]
        filtered_pnl_df = df[common_cols].copy()

        return PnLAnalyzer(filtered_pnl_df, self.position_df)

    def get_position_index(self) -> list:
        if 'position_index' not in self._pos_and_pnl_df.columns:
            print("[WARN] 'position_index' column not found.")
            return []
        return self._pos_and_pnl_df['position_index'].dropna().unique().tolist()

    def pivot(self, index='unit', columns='pnl_date', values='lookback_pnl'):
        df = self._pos_and_pnl_df

        # Flatten index, columns, and values to lists for uniform checking
        index_cols = index if isinstance(index, (list, tuple)) else [index]
        column_cols = columns if isinstance(columns, (list, tuple)) else [columns]
        value_cols = [values]  # always a single column

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
