from abc import ABC, abstractmethod



class PriceLoader(ABC):

    def __init__(self, instrument_name, source, params=None):
        self.instrument_id = instrument_name
        self.source = source
        self.params = params or {}

    @abstractmethod
    def load_prices(self, start_date, end_date):
        """Load historical price_series_loader data into price_history."""
        print('load price_series_loader')
        pass
