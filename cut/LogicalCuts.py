from .CutBase import UnbinnedCutBase
from typing import Any, List, Sequence, Union
from simonplot.typing.Protocols import CutProtocol, VariableProtocol, UnbinnedDatasetAccessProtocol, UnbinnedDatasetProtocol
import numpy as np

class AndCuts(UnbinnedCutBase):
    def __init__(self, cuts : Sequence[CutProtocol]):
        self._cuts = cuts

    @property
    def columns(self):
        cols = []
        for cut in self._cuts:
            cols += cut.columns
        return list(set(cols))

    def evaluate(self, dataset):
        dataset = self.ensure_valid_dataset(dataset)   

        mask = self._cuts[0].evaluate(dataset)
        if isinstance(mask, slice): #ensure mask is a boolean array
            mask = np.ones(dataset.num_rows, dtype=bool)[mask]

        for cut in self._cuts[1:]:
            nextmask = cut.evaluate(dataset)
            if isinstance(mask, slice): #ensure mask is a boolean array
                nextmask = np.ones(dataset.num_rows, dtype=bool)[nextmask]
            
            mask = np.logical_and(mask, nextmask)

        return mask

    @property
    def key(self):
        result = self._cuts[0].key
        for cut in self._cuts[1:]:
            result += "_AND_" + cut.key
        return result

    @property
    def _auto_label(self):
        result = self._cuts[0].label
        for cut in self._cuts[1:]:
            result += '\n' + cut.label
        return result

    def set_collection_name(self, collection_name):
        for cut in self._cuts:
            cut.set_collection_name(collection_name)

    def __eq__(self, other):
        if not isinstance(other, AndCuts):
            return False
        
        if len(self._cuts) != len(other._cuts):
            return False
        
        for cut in self._cuts:
            found = False
            for other_cut in other._cuts:
                if cut == other_cut:
                    found = True
                    break
            if not found:
                return False
        
        return True

class OrCuts(UnbinnedCutBase):
    def __init__(self, cuts : Sequence[CutProtocol]):
        self._cuts = cuts

    @property
    def columns(self):
        cols = []
        for cut in self._cuts:
            cols += cut.columns
        return list(set(cols))

    def evaluate(self, dataset):
        dataset = self.ensure_valid_dataset(dataset)   

        mask = self._cuts[0].evaluate(dataset)
        if isinstance(mask, slice): #ensure mask is a boolean array
            mask = np.ones(dataset.num_rows, dtype=bool)[mask]

        for cut in self._cuts[1:]:
            nextmask = cut.evaluate(dataset)
            if isinstance(mask, slice): #ensure mask is a boolean array
                nextmask = np.ones(dataset.num_rows, dtype=bool)[nextmask]
            
            mask = np.logical_or(mask, nextmask)

        return mask

    @property
    def key(self):
        result = self._cuts[0].key
        for cut in self._cuts[1:]:
            result += "_OR_" + cut.key
        return result

    @property
    def _auto_label(self):
        result = self._cuts[0].label
        for cut in self._cuts[1:]:
            result += '\nOR ' + cut.label
        return result

    def set_collection_name(self, collection_name):
        for cut in self._cuts:
            cut.set_collection_name(collection_name)

    def __eq__(self, other):
        if not isinstance(other, OrCuts):
            return False
        
        if len(self._cuts) != len(other._cuts):
            return False
        
        for cut in self._cuts:
            found = False
            for other_cut in other._cuts:
                if cut == other_cut:
                    found = True
                    break
            if not found:
                return False
        
        return True


class NotCut(UnbinnedCutBase):
    def __init__(self, cut : CutProtocol):
        print("Warning: NotCut is discouraged because the autmatic labels aren't as pretty")
        print("When possible, phrase the cut in terms of positive logic instead")
        self._cut = cut

    @property
    def columns(self):
        return self._cut.columns

    def evaluate(self, dataset):
        dataset = self.ensure_valid_dataset(dataset)    

        mask = self._cut.evaluate(dataset)
        if isinstance(mask, slice):
            mask = np.ones(dataset.num_rows, dtype=bool)[mask]

        return np.logical_not(mask)

    @property
    def key(self):
        return "NOT(%s)"%(self._cut.key)

    @property
    def _auto_label(self):
        return "NOT (%s)"%(self._cut.label)

    def set_collection_name(self, collection_name):
        self._cut.set_collection_name(collection_name)

    def __eq__(self, other):
        if not isinstance(other, NotCut):
            return False

        return self._cut == other._cut