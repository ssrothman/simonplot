import numpy as np
import copy

from typing import Any, List, Sequence, Union

from numpy._typing._array_like import NDArray

from simonplot.variable.Variable import BasicVariable

from .CutBase import UnbinnedCutBase
from simonplot.typing.Protocols import CutProtocol, VariableProtocol, UnbinnedDatasetAccessProtocol, UnbinnedDatasetProtocol

from .NoCut import NoCut

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
