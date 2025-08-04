from abc import ABC


class LoadedPosition(ABC):

    def __init__(self, source, params=None):
        self.source = source
        self.params = params or {}
