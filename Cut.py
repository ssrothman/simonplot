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
        return "%s$ = %g$"%(config['axis_labels'][self.variable.key], 
                            self.value)

class TwoSidedCut(AbstractCut):
    def __init__(self, variable, low, high):
        self.low = low
        self.high = high
        if type(variable) is str:
            self.variable = variable_from_string(variable)
        else:
            self.variable = variable

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
                config['axis_labels'][self.variable.key], 
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
        return "%s$ \\geq %g$"%(config['axis_labels'][self.variable.key], 
                            self.value)
    
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
        return "%s$ < %g$"%(config['axis_labels'][self.variable.key], 
                            self.value)

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
