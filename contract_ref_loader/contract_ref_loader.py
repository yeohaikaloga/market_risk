# For reference, e.g. contract_ref_loader: 'CT'; contract_ref_loader: 'CTH4'
from abc import ABC, abstractmethod


class ContractRefLoader(ABC):
    def __init__(self, instrument_name, source, params=None):
        self.instrument_name = instrument_name
        self.source = source
        self.params = params or {}
