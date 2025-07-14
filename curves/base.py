from abc import ABC, abstractmethod
import pandas as pd


class BaseCurveConstructor(ABC):
    def __init__(self, price_df: pd.DataFrame):
        self.price_df = price_df.sort_index()
        self.validate_data()
        self.contracts = self.price_df.columns.tolist()

    def validate_data(self):
        if self.price_df.empty:
            raise ValueError("Price DataFrame is empty.")

    @abstractmethod
    def construct_curve(self) -> pd.Series:
        """
        Subclasses must implement this method to generate the curve.
        """
        pass

    def get_active_contracts(self, date):
        row = self.price_df.loc[date]
        return row[row.notna()]

    def calculate_roll_dates(self, roll_days: int) -> dict:
        roll_dates = {}
        for contract in self.contracts:
            valid_dates = self.price_df[contract].dropna().index
            if not valid_dates.empty:
                last_date = valid_dates[-1]
                roll_idx = max(0, len(valid_dates) - 1 - roll_days)
                roll_dates[contract] = valid_dates[roll_idx]
        return roll_dates
