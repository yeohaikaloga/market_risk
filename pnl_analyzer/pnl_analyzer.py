import pandas as pd
import numpy as np


class PnLAnalyzer:
    def __init__(self, pnl_df: pd.DataFrame, position_df: pd.DataFrame = None):
        self.pnl_df = pnl_df.copy()
        self.position_df = position_df.copy() if position_df is not None else None
        self._pos_and_pnl_df = self._merge_pos_and_pnl() if position_df is not None else pnl_df

    def _merge_pos_and_pnl(self):
        return self.pnl_df.merge(self.position_df, on='position_index', how='left')

    def filter(self, **kwargs):
        df = self._pos_and_pnl_df.copy()  # use merged df
        for key, value in kwargs.items():
            if isinstance(value, (list, set, tuple, pd.Series, np.ndarray)):
                df = df[df[key].isin(value)]
            else:
                df = df[df[key] == value]
        common_cols = [col for col in self.pnl_df.columns if col in df.columns]
        filtered_pnl_df = df[common_cols].copy()
        return PnLAnalyzer(filtered_pnl_df, self.position_df)

    def pivot(self, index='unit', columns='pnl_date', values='lookback_pnl'):
        df = self._pos_and_pnl_df
        missing_keys = [key for key in [index, columns, values] if key not in df.columns]
        if missing_keys:
            print("Available columns:", df.columns.tolist())
            raise KeyError(f"Missing pivot key(s): {missing_keys}")

        return df.pivot_table(index=index, columns=columns, values=values, aggfunc='sum')

    def to_excel(self, path):
        # Example: Export grouped pivot or raw data
        with pd.ExcelWriter(path) as writer:
            self._pos_and_pnl_df.to_excel(writer, sheet_name='PnL')
            self.pivot().to_excel(writer, sheet_name='Pivot')
