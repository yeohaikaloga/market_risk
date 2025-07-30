from abc import ABC, abstractmethod
import pandas as pd


class LoadedPrice(ABC):

    def __init__(self, instrument_id, source, params=None):
        self.instrument_id = instrument_id
        self.source = source
        self.params = params or {}

    @abstractmethod
    def load_prices(self, start_date, end_date):
        """Load historical loaded_price_series data into price_history."""
        print('load loaded_price_series')
        pass
