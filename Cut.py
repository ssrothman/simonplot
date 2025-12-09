from .Variable import variable_from_string
from .SetupConfig import config
import numpy as np

class AbstractCut:
    @property
    def columns(self):
        raise NotImplementedError()

    def evaluate(self, table):
        raise NotImplementedError()

    @property
    def key(self):
        raise NotImplementedError()

    @property
    def plottext(self):
        raise NotImplementedError()
    
    #equality operator
    def __eq__(self, other):
        return False

class NoCut(AbstractCut):
    def __init__(self):
        pass

    @property
    def columns(self):
        return []

    def evaluate(self, table):
        return np.ones(table.num_rows, dtype=bool)

    @property
    def key(self):
        return "none"

    @property
    def plottext(self):
        return "Inclusive"

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

    def evaluate(self, table):
        ev = self.variable.evaluate(table)
        return ev == self.value

    @property
    def key(self):
        return "%sEQ%g"%(self.variable.key, self.value)

    @property
    def plottext(self):
        return "%s$ = %g$"%(self.variable.label, 
                            self.value)
    
    def __eq__(self, other):
        if type(other) is not EqualsCut:
            return False
        return self.variable.key == other.variable.key and self.value == other.value

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
        return (self.variable.key == other.variable.key and
                self.low == other.low and
                self.high == other.high)

    @property
    def columns(self):
        return self.variable.columns

    def evaluate(self, table):
        ev = self.variable.evaluate(table)
        return np.logical_and(
            ev >= self.low,
            ev < self.high
        )

    @property
    def key(self):
        return "%sLT%gGT%g"%(self.variable.key, self.high, self.low)

    @property
    def plottext(self):
        return "$%g \\leq $%s$ < %g$"%(
                self.low,
                self.variable.label, 
                self.high)
    
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

    def evaluate(self, table):
        ev = self.variable.evaluate(table)
        return ev >= self.value

    @property
    def key(self):
        return "%sGT%g"%(self.variable.key, self.value)

    @property
    def plottext(self):
        return "%s$ \\geq %g$"%(self.variable.label, 
                            self.value)
    
    def __eq__(self, other):
        if type(other) is not GreaterThanCut:
            return False
        return self.variable.key == other.variable.key and self.value == other.value

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

    def evaluate(self, table):
        ev = self.variable.evaluate(table)
        return ev < self.value

    @property
    def key(self):
        return "%sLT%g"%(self.variable.key, self.value)

    @property
    def plottext(self):
        return "%s$ < %g$"%(self.variable.label, 
                            self.value)

    def __eq__(self, other):
        if type(other) is not LessThanCut:
            return False
        return self.variable.key == other.variable.key and self.value == other.value

class AndCuts(AbstractCut):
    def __init__(self, *cuts):
        self.cuts = cuts

    @property
    def columns(self):
        cols = []
        for cut in self.cuts:
            cols += cut.columns
        return list(set(cols))

    def evaluate(self, table):
        mask = self.cuts[0].evaluate(table)
        for cut in self.cuts[1:]:
            mask = np.logical_and(mask, cut.evaluate(table))
        return mask

    @property
    def key(self):
        result = self.cuts[0].key
        for cut in self.cuts[1:]:
            result += "_AND_" + cut.key
        return result

    @property
    def plottext(self):
        result = self.cuts[0].plottext
        for cut in self.cuts[1:]:
            result += '\n' + cut.plottext
        return result

def common_cuts_(cut1, cut2):
    if type(cut1) is AndCuts and type(cut2) is not AndCuts:
        return common_cuts(list(cut1.cuts) + [cut2])
    elif type(cut2) is AndCuts and type(cut1) is not AndCuts:
        return common_cuts([cut1] + list(cut2.cuts))
    elif type(cut1) is AndCuts and type(cut2) is AndCuts:
        c1s = list(cut1.cuts)
        c2s = list(cut2.cuts)

        common = []
        for c1 in c1s:
            for c2 in c2s:
                if c1 == c2:
                    common.append(c1)
        
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
