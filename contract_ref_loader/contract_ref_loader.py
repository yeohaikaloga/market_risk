# For reference, e.g. contract_ref_loader: 'CT'; contract_ref_loader: 'CTH4'
from abc import ABC, abstractmethod
import pandas as pd


class Contract(ABC):
    def __init__(self, instrument_id, source, params=None):
        self.instrument_id = instrument_id
        self.source = source
        self.params = params or {}

    @abstractmethod
    def load_ref_data(self):
        """Load static metadata for the contract_ref_loader."""
        pass

    @abstractmethod
    def _load_contract_data(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def load_contracts(self):
        """Load list of contracts."""
        pass

    @abstractmethod
    def load_expiry_dates(self, start_date, end_date):
        """Load list of expiry dates for active contracts for the contract_ref_loader within a date range."""
        pass
