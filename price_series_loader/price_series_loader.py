from abc import ABC, abstractmethod



class LoadedPrice(ABC):

    def __init__(self, instrument_id, source, params=None):
        self.instrument_id = instrument_id
        self.source = source
        self.params = params or {}

    @abstractmethod
    def load_prices(self, start_date, end_date):
        """Load historical price_series_loader data into price_history."""
        print('load price_series_loader')
        pass
