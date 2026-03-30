
from .CutBase import UnbinnedCutBase

class NoCut(UnbinnedCutBase):
    def __init__(self):
        pass

    @property
    def columns(self):
        return []

    def evaluate(self, dataset):
        return slice(None)
        
    @property
    def key(self):
        return "none"

    @property
    def _auto_label(self):
        return ""

    def __eq__(self, other):
        return False 
    
    def set_collection_name(self, collection_name):
        pass