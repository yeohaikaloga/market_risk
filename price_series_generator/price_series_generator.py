from abc import ABC
import pandas as pd


class PriceSeriesGenerator(ABC):
    def __init__(self, price_df: pd.DataFrame):
        self.price_df = price_df.sort_index()
        self.contracts = self.price_df.columns.tolist()
