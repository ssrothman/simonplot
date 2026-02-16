import numpy as np
import copy

from typing import Any, List, Sequence, Union

from numpy._typing._array_like import NDArray

from simonplot.variable.Variable import BasicVariable

from .CutBase import UnbinnedCutBase
from simonplot.typing.Protocols import CutProtocol, VariableProtocol, UnbinnedDatasetAccessProtocol, UnbinnedDatasetProtocol

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
        return "Inclusive"

    def __eq__(self, other):
        return False 
    
    def set_collection_name(self, collection_name):
        pass

class EqualsCut(UnbinnedCutBase):
    def __init__(self, variable : VariableProtocol | str, value : float | int):
        self._value = value
        if isinstance(variable, str):
            self._variable = BasicVariable(variable)
        else:
            self._variable = variable

    @property
    def columns(self):
        return self._variable.columns

    def evaluate(self, dataset):
        dataset = self.ensure_valid_dataset(dataset)
        ev = self._variable.evaluate(dataset, NoCut())
        return ev == self._value

    @property
    def key(self):
        return "%sEQ%g"%(self._variable.key, self._value)

    @property
    def _auto_label(self):
        return "%s$ = %g$"%(self._variable.label, 
                            self._value)
    
    def __eq__(self, other):
        if not isinstance(other, EqualsCut):
            return False
        
        return self._variable == other._variable and self._value == other._value

    def set_collection_name(self, collection_name):
        self._variable.set_collection_name(collection_name)

class AllEqualCut(UnbinnedCutBase):
    def __init__(self, variables : List[VariableProtocol | str], value : float | int):
        self._value = value
        self._variables = [BasicVariable(var) if isinstance(var, str) else var for var in variables]

    @property
    def columns(self):
        cols = []
        for var in self._variables:
            cols += var.columns
        return list(set(cols))
    
    def evaluate(self, dataset):
        dataset = self.ensure_valid_dataset(dataset)
        mask = np.ones(dataset.num_rows, dtype=bool)
        for var in self._variables:
            ev = var.evaluate(dataset, NoCut())
            mask = np.logical_and(mask, ev == self._value)

        return mask
    
    @property
    def key(self):
        varkey = '_'.join([var.key for var in self._variables])
        return "%sEQ%g"%(varkey, self._value)
    
    @property
    def _auto_label(self):
        varlabels = ' = '.join([var.label for var in self._variables])
        return "%s$ = %g$"%(varlabels, self._value)
    
    def __eq__(self, other):
        if not isinstance(other, AllEqualCut):
            return False
        
        if self._value != other._value:
            return False
        
        if len(self._variables) != len(other._variables):
            return False
        
        for var in self._variables:
            found = False
            for other_var in other._variables:
                if var == other_var:
                    found = True
                    break
            if not found:
                return False
        
        return True
    
    def set_collection_name(self, collection_name):
        for var in self._variables:
            var.set_collection_name(collection_name)

class TwoSidedCut(UnbinnedCutBase):
    def __init__(self, variable : VariableProtocol | str, low : float | int, high : float | int):
        self._low = low
        self._high = high
        if isinstance(variable, str):
            self._variable = BasicVariable(variable)
        else:
            self._variable = variable

    def __eq__(self, other):
        if not isinstance(other, TwoSidedCut):  
            return False
        
        return (self._variable == other._variable and
                self._low == other._low and
                self._high == other._high)

    @property
    def columns(self):
        return self._variable.columns

    def evaluate(self, dataset):
        dataset = self.ensure_valid_dataset(dataset)
        ev = self._variable.evaluate(dataset, NoCut())
        return np.logical_and(
            ev >= self._low,
            ev < self._high
        )

    @property
    def key(self):
        return "%sLT%gGT%g"%(self._variable.key, self._high, self._low)

    @property
    def _auto_label(self):
        return "$%g \\leq $%s$ < %g$"%(
                self._low,
                self._variable.label, 
                self._high)
    
    def set_collection_name(self, collection_name):
        self._variable.set_collection_name(collection_name)

class GreaterThanCut(UnbinnedCutBase):
    def __init__(self, variable : VariableProtocol | str , value : int | float):
        self._value = value
        if isinstance(variable, str):
            self._variable = BasicVariable(variable)
        else:
            self._variable = variable

    @property
    def columns(self):
        return self._variable.columns

    def evaluate(self, dataset):
        dataset = self.ensure_valid_dataset(dataset)
        ev = self._variable.evaluate(dataset, NoCut())
        return ev >= self._value

    @property
    def key(self):
        return "%sGT%g"%(self._variable.key, self._value)

    @property
    def _auto_label(self):
        return "%s$ \\geq %g$"%(self._variable.label, 
                            self._value)
    
    def __eq__(self, other):
        if not isinstance(other, GreaterThanCut):
            return False
        return self._variable == other._variable and self._value == other._value

    def set_collection_name(self, collection_name):
        self._variable.set_collection_name(collection_name)

class LessThanCut(UnbinnedCutBase):
    def __init__(self, variable : VariableProtocol | str, value : int | float):
        self._value = value
        if isinstance(variable, str):
            self._variable = BasicVariable(variable)
        else:
            self._variable = variable

    @property
    def columns(self):
        return self._variable.columns

    def evaluate(self, dataset):
        dataset = self.ensure_valid_dataset(dataset)
        ev = self._variable.evaluate(dataset, NoCut())
        return ev < self._value

    @property
    def key(self):
        return "%sLT%g"%(self._variable.key, self._value)

    @property
    def _auto_label(self):
        return "%s$ < %g$"%(self._variable.label, 
                            self._value)

    def __eq__(self, other):
        if not isinstance(other, LessThanCut):
            return False
        
        return self._variable == other._variable and self._value == other._value

    def set_collection_name(self, collection_name):
        self._variable.set_collection_name(collection_name)

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

class ConcatCut(UnbinnedCutBase):
    def __init__(self, *cuts, keycut=None):
        self.cuts = cuts

        if keycut is None:
            self._keycut = NoCut()
            print("Warning: ConcatCut has no keycut, so automatic labels will be blank")
        else:
            self._keycut = keycut

    @staticmethod
    def build_for_collections(cut : CutProtocol, collections_l : Sequence[str], unique_cuts_l : Union[None, Sequence[CutProtocol]]=None):
        if unique_cuts_l is not None and len(unique_cuts_l) != len(collections_l):
            raise ValueError("ConcatCut.build_for_collections: unique_cuts_l length does not match collections_l length")
        
        if unique_cuts_l is None:
            unique_cuts_l = [NoCut()] * len(collections_l)

        cuts : List[CutProtocol] = []
        for coll, ucut in zip(collections_l, unique_cuts_l):
            c = copy.deepcopy(cut)
            c.set_collection_name(coll)
            cuts.append(AndCuts([c, ucut]))
            
        return ConcatCut(*cuts, keycut=cut)

    @property
    def columns(self):
        cols = []
        for cut in self.cuts:
            cols += cut.columns
        return list(set(cols))

    @property
    def keycut(self):
        return self._keycut

    def evaluate(self, dataset):
        dataset = self.ensure_valid_dataset(dataset)   
        masks = [cut.evaluate(dataset) for cut in self.cuts]
        return np.concatenate(masks)

    @property
    def key(self):
        return self._keycut.key

    @property
    def _auto_label(self):
        return self._keycut.label

    def set_collection_name(self, collection_name):
        print("WARNING: overwriting collection name for all cuts in ConcatCut object")
        for cut in self.cuts:
            cut.set_collection_name(collection_name)
        self._keycut.set_collection_name(collection_name)

    def __eq__(self, other):
        if not isinstance(other, ConcatCut):
            return False

        return self._keycut == other._keycut
