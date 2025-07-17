# For reference, e.g. contract: 'CT'; contract: 'CTH4'
from abc import ABC, abstractmethod


class Contract(ABC):
    def __init__(self, instrument_id, source):
        self.instrument_id = instrument_id  # e.g., "CT"
        self.source = source                # SQLAlchemy engine or connection
        self.currency_id = None
        self.unit = None
        self.lot_size = None
        self.exchange = None
        self.active_contracts = []

    @abstractmethod
    def load_ref_data(self):
        """Load static metadata for the contract."""
        pass

    @abstractmethod
    def load_contracts(self, start_date, end_date):
        """Load list of active contracts for the contract within a date range."""
        pass

    @abstractmethod
    def load_expiry_dates(self, start_date, end_date):
        """Load list of expiry dates for active contracts for the contract within a date range."""
        pass
