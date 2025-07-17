from abc import ABC, abstractmethod
import pandas as pd


class Price(ABC):

    def __init__(self, instrument_id, currency_id=None, unit=None, lot_size=None, source=None, location=None,
                 exchange=None, active_tickers=None):
        self.instrument_id = instrument_id
        self.currency_id = currency_id
        self.unit = unit
        self.lot_size = lot_size
        self.source = source
        self.location = location
        self.exchange = exchange
        self.active_tickers = active_tickers
        self.price_history = pd.DataFrame()

    @abstractmethod
    def load_prices(self, start_date, end_date):
        """Load historical price data into price_history."""
        print('load price')
        pass

    def latest_price(self):
        return self.price_history.iloc[-1] if not self.price_history.empty else None
