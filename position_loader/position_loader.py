from abc import ABC


class PositionLoader(ABC):

    def __init__(self, source, params=None):
        self.source = source
        self.params = params or {}
