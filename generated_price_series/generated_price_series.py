from abc import ABC, abstractmethod
import pandas as pd


class PriceSeriesGenerator(ABC):
    def __init__(self, price_df: pd.DataFrame):
        self.price_df = price_df.sort_index()
        self.validate_data()
        self.contracts = self.price_df.columns.tolist()

    @abstractmethod
    def validate_data(self):
        pass

    @abstractmethod
    def get_active_contracts(self, date):
        pass
