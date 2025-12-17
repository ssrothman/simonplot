from .Variable import variable_from_string
from .SetupConfig import config
import numpy as np
import copy
from typing import List, Sequence, Union

class AbstractCut:
    @property
    def columns(self):
        raise NotImplementedError()

    def evaluate(self, dataset):
        raise NotImplementedError()

    @property
    def key(self):
        raise NotImplementedError()

    @property
    def plottext(self):
        if hasattr(self, "_plottext"):
            return self._plottext
        else:
            return self._auto_plottext()

    def _auto_plottext(self):
        raise NotImplementedError()

    def override_plottext(self, plottext):
        self._plottext = plottext

    def clear_override_plottext(self):
        del self._plottext

    #equality operator
    def __eq__(self, other):
        #error message says what subclass raised the error
        raise NotImplementedError("Equality operator not implemented for subclass %s"%(type(self).__name__))
    
    def set_collection_name(self, collection_name):
        raise NotImplementedError()

class NoCut(AbstractCut):
    def __init__(self):
        pass

    @property
    def columns(self):
        return []

    def evaluate(self, dataset):
        return np.ones(dataset.num_rows, dtype=bool)

    @property
    def key(self):
        return "none"

    def _auto_plottext(self):
        return "Inclusive"

    def __eq__(self, other):
        return False 
    
    def set_collection_name(self, collection_name):
        pass

class EqualsCut(AbstractCut):
    def __init__(self, variable, value):
        self.value = value
        if type(variable) is str:
            self.variable = variable_from_string(variable)
        else:
            self.variable = variable

    @property
    def columns(self):
        return self.variable.columns

    def evaluate(self, dataset):
        ev = self.variable.evaluate(dataset)
        return ev == self.value

    @property
    def key(self):
        return "%sEQ%g"%(self.variable.key, self.value)

    def _auto_plottext(self):
        return "%s$ = %g$"%(self.variable.label, 
                            self.value)
    
    def __eq__(self, other):
        if type(other) is not EqualsCut:
            return False
        return self.variable == other.variable and self.value == other.value

    def set_collection_name(self, collection_name):
        self.variable.set_collection_name(collection_name)

class TwoSidedCut(AbstractCut):
    def __init__(self, variable, low, high):
        self.low = low
        self.high = high
        if type(variable) is str:
            self.variable = variable_from_string(variable)
        else:
            self.variable = variable

    def __eq__(self, other):
        if type(other) is not TwoSidedCut:
            return False
        return (self.variable == other.variable and
                self.low == other.low and
                self.high == other.high)

    @property
    def columns(self):
        return self.variable.columns

    def evaluate(self, dataset):
        ev = self.variable.evaluate(dataset)
        return np.logical_and(
            ev >= self.low,
            ev < self.high
        )

    @property
    def key(self):
        return "%sLT%gGT%g"%(self.variable.key, self.high, self.low)

    def _auto_plottext(self):
        return "$%g \\leq $%s$ < %g$"%(
                self.low,
                self.variable.label, 
                self.high)
    
    def set_collection_name(self, collection_name):
        self.variable.set_collection_name(collection_name)

class GreaterThanCut(AbstractCut):
    def __init__(self, variable, value):
        self.value = value
        if type(variable) is str:
            self.variable = variable_from_string(variable)
        else:
            self.variable = variable

    @property
    def columns(self):
        return self.variable.columns

    def evaluate(self, dataset):
        ev = self.variable.evaluate(dataset)
        return ev >= self.value

    @property
    def key(self):
        return "%sGT%g"%(self.variable.key, self.value)

    def _auto_plottext(self):
        return "%s$ \\geq %g$"%(self.variable.label, 
                            self.value)
    
    def __eq__(self, other):
        if type(other) is not GreaterThanCut:
            return False
        return self.variable == other.variable and self.value == other.value

    def set_collection_name(self, collection_name):
        self.variable.set_collection_name(collection_name)

class LessThanCut(AbstractCut):
    def __init__(self, variable, value):
        self.value = value
        if type(variable) is str:
            self.variable = variable_from_string(variable)
        else:
            self.variable = variable

    @property
    def columns(self):
        return self.variable.columns

    def evaluate(self, dataset):
        ev = self.variable.evaluate(dataset)
        return ev < self.value

    @property
    def key(self):
        return "%sLT%g"%(self.variable.key, self.value)

    def _auto_plottext(self):
        return "%s$ < %g$"%(self.variable.label, 
                            self.value)

    def __eq__(self, other):
        if type(other) is not LessThanCut:
            return False
        return self.variable == other.variable and self.value == other.value

    def set_collection_name(self, collection_name):
        self.variable.set_collection_name(collection_name)

class AndCuts(AbstractCut):
    def __init__(self, *cuts):
        self.cuts = cuts

    @property
    def columns(self):
        cols = []
        for cut in self.cuts:
            cols += cut.columns
        return list(set(cols))

    def evaluate(self, dataset):
        mask = self.cuts[0].evaluate(dataset)
        for cut in self.cuts[1:]:
            mask = np.logical_and(mask, cut.evaluate(dataset))
        return mask

    @property
    def key(self):
        result = self.cuts[0].key
        for cut in self.cuts[1:]:
            result += "_AND_" + cut.key
        return result

    def _auto_plottext(self):
        result = self.cuts[0].plottext
        for cut in self.cuts[1:]:
            result += '\n' + cut.plottext
        return result

    def set_collection_name(self, collection_name):
        for cut in self.cuts:
            cut.set_collection_name(collection_name)

    def __eq__(self, other):
        if type(other) is not AndCuts:
            return False
        
        if len(self.cuts) != len(other.cuts):
            return False
        
        for cut in self.cuts:
            found = False
            for other_cut in other.cuts:
                if cut == other_cut:
                    found = True
                    break
            if not found:
                return False
        
        return True

class ConcatCut(AbstractCut):
    def __init__(self, *cuts, keycut=None):
        self.cuts = cuts

        if keycut is None:
            self._keycut = NoCut()
            print("Warning: ConcatCut has no keycut, so automatic labels will be blank")
        else:
            self._keycut = keycut

    @staticmethod
    def build_for_collections(cut : AbstractCut, collections_l : List[str], unique_cuts_l : Union[None, Sequence[AbstractCut]]=None):
        if unique_cuts_l is not None and len(unique_cuts_l) != len(collections_l):
            raise ValueError("ConcatCut.build_for_collections: unique_cuts_l length does not match collections_l length")
        
        if unique_cuts_l is None:
            unique_cuts_l = [NoCut()] * len(collections_l)

        cuts = []
        for coll, ucut in zip(collections_l, unique_cuts_l):
            c = copy.deepcopy(cut)
            c.set_collection_name(coll)
            cuts.append(AndCuts(c, ucut))
            
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
        masks = [cut.evaluate(dataset) for cut in self.cuts]
        return np.concatenate(masks)

    @property
    def key(self):
        return self._keycut.key

    def _auto_plottext(self):
        return self._keycut.plottext

    def set_collection_name(self, collection_name):
        print("WARNING: overwriting collection name for all cuts in ConcatCut object")
        for cut in self.cuts:
            cut.set_collection_name(collection_name)

    def __eq__(self, other):
        if type(other) is not ConcatCut:
            return False
        return self._keycut == other._keycut

def common_cuts_(cut1, cut2):
    if type(cut1) is ConcatCut:
        cut1 = cut1.keycut
    if type(cut2) is ConcatCut:
        cut2 = cut2.keycut

    if type(cut1) is AndCuts and type(cut2) is not AndCuts:
        common = []
        for c1 in cut1.cuts:
            if c1 == cut2:
                common.append(c1)
        if len(common) == 0:
            return NoCut()
        elif len(common) == 1:
            return common[0]
        else:
            return AndCuts(*common)
        
    elif type(cut2) is AndCuts and type(cut1) is not AndCuts:
        return common_cuts_(cut2, cut1)
    
    elif type(cut1) is AndCuts and type(cut2) is AndCuts:
        c1s = list(cut1.cuts)
        c2s = list(cut2.cuts)

        common = []
        for c1 in c1s:
            for c2 in c2s:
                if c1 == c2:
                    common.append(c1)
        
        if len(common) == 0:
            return NoCut()
        elif len(common) == 1:
            return common[0]
        else:
            return AndCuts(*common)

    elif cut1 == cut2:
        return cut1
    else:  
        return NoCut()

def common_cuts(cuts):
    if len(cuts) == 0:
        return NoCut()
    elif len(cuts) == 1:
        return cuts[0]
    else:
        common = cuts[0]
        for cut in cuts[1:]:
            common = common_cuts_(common, cut)
        return common
