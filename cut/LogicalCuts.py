from .CutBase import UnbinnedCutBase
from typing import Any, List, Sequence, Union
from simonplot.typing.Protocols import CutProtocol, VariableProtocol, UnbinnedDatasetAccessProtocol, UnbinnedDatasetProtocol
import numpy as np
from .NoCut import NoCut

def get_cuts_list(cuts : Union[CutProtocol, Sequence[CutProtocol]]):
    if isinstance(cuts, NoCut):
        return []
    elif hasattr(cuts, '_cuts'): # AndCuts and OrCuts have a _cuts attribute which is a list of their component cuts
        return get_cuts_list(getattr(cuts, '_cuts'))
    elif isinstance(cuts, CutProtocol):
        return [cuts]
    elif isinstance(cuts, Sequence):
        result = []
        for cut in cuts:
            result += get_cuts_list(cut)
        return result
    else:
        assert_never(cuts)

class AndCuts(UnbinnedCutBase):
    # get in before __init__ and sometimes return a different class
    def __new__(cls, cuts : Sequence[CutProtocol]):
        filtered_cuts = get_cuts_list(cuts)

        if len(filtered_cuts) == 0:
            return NoCut()
        elif len(filtered_cuts) == 1:
            return filtered_cuts[0]
        else:
            # Build the AndCuts object; __init__ will run automatically.
            return super(AndCuts, cls).__new__(cls)

    def __init__(self, cuts : Sequence[CutProtocol]):
        filtered_cuts = get_cuts_list(cuts)
        self._cuts = filtered_cuts

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
    # get in before __init__ and sometimes return a different class
    def __new__(cls, cuts : Sequence[CutProtocol]):
        filtered_cuts = get_cuts_list(cuts)

        if len(filtered_cuts) == 0:
            return NoCut()
        elif len(filtered_cuts) == 1:
            return filtered_cuts[0]
        else:
            # Build the OrCuts object; __init__ will run automatically.
            return super(OrCuts, cls).__new__(cls)

    def __init__(self, cuts : Sequence[CutProtocol]):
        self._cuts = get_cuts_list(cuts)

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