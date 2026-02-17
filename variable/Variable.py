import copy

from simonplot.config import lookup_axis_label
from .VariableBase import VariableBase

from typing import List, Sequence, assert_never, override
import awkward as ak
import numpy as np

from simonplot.typing.Protocols import VariableProtocol

class ConstantVariable(VariableBase):
    def __init__(self, value : float | int):
        self._value = value

    @property
    def _natural_centerline(self):
        return None
    
    @property
    def prebinned(self) -> bool:
        return False
    
    @property
    def columns(self):
        return []
    
    def evaluate(self, dataset, cut):
        mask = cut.evaluate(dataset)
        if isinstance(mask, ak.Array):
            val = ak.ones_like(mask) * self._value
        elif isinstance(mask, np.ndarray):
            val = np.ones_like(mask) * self._value
        elif isinstance(mask, slice):
            val = np.array([self._value])
            val = np.repeat(val, dataset.num_rows)
        else: 
            assert_never(mask)

        return val[mask]
    
    @property
    def key(self):
        return "CONST(%s)"%(self._value)
        
    def set_collection_name(self, collection_name):
        pass #no-op

    def __eq__(self, other):
        if type(other) is not ConstantVariable:
            return False
        
        return self._value == other._value

class BasicVariable(VariableBase):
    def __init__(self, name : str, collection_name: str | None = None):
        self._name = name
        self._collection_name = collection_name

        if "." in name and collection_name is None:
            splitted = name.split(".")
            self._collection_name = ".".join(splitted[:-1])
            self._name = splitted[-1]
        elif '.' in name:
            raise ValueError("Variable name '%s' contains '.' but collection_name is also specified"%(name))

    @property
    def _natural_centerline(self):
        return None
    
    @property
    def columns(self):
        if self._collection_name is None:
            return [self._name]
        else:
            return [self._collection_name + "." + self._name]

    def evaluate(self, dataset, cut):
        mask = cut.evaluate(dataset)
        val = dataset.get_column(self._name, self._collection_name)
        return val[mask]
    
    @property
    def prebinned(self) -> bool:
        return False
    
    @property
    def key(self):
        if self._collection_name is None:
            return self._name
        else:
            return self._collection_name + "." + self._name
    
    def __eq__(self, other):
        if type(other) is not BasicVariable:
            return False
        
        #NB do NOT require collection name to match
        #This allows e.g. ECALRecHit.x == HCALRecHit.x [desireable]
        #But also potentially perverse situations like ECALRecHit.x == GenVertex.x [conceptually distinct?]
        #No good solution to this currently
        return self._name == other._name 
    
    def set_collection_name(self, collection_name):
        self._collection_name = collection_name

class AkNumVariable(VariableBase):
    def __init__(self, var : VariableProtocol | str):
        self._var = BasicVariable(var) if isinstance(var, str) else var

    @property
    def _natural_centerline(self):
        return None
    
    @property
    def columns(self):
        return self._var.columns

    @property
    def prebinned(self) -> bool:
        return False
    
    def evaluate(self, dataset, cut):
        val = self._var.evaluate(dataset, cut)
        mask = cut.evaluate(dataset)
        return ak.num(val[mask])
    
    @property
    def key(self):
        return "N(%s)"%(self._var.key)
    
    def __eq__(self, other):
        if type(other) is not AkNumVariable:
            return False
        
        return self._var == other._var
    
    def set_collection_name(self, collection_name):
        raise ValueError("AkNumVariable does not support set_collection_name")

class RatioVariable(VariableBase):
    def __init__(self, num : VariableProtocol | str, denom : VariableProtocol | str):
        self._num = BasicVariable(num) if isinstance(num, str) else num
        self._denom = BasicVariable(denom) if isinstance(denom, str) else denom

    @property
    def _natural_centerline(self):
        return 1.0
    
    @property
    def prebinned(self) -> bool:
        return False
    
    @property
    def columns(self):
        return list(set(self._num.columns + self._denom.columns))

    def evaluate(self, dataset, cut):
        return self._num.evaluate(dataset, cut) / self._denom.evaluate(dataset, cut)

    @property
    def key(self):
        return "%s_over_%s"%(self._num.key, self._denom.key)
    
    def __eq__(self, other):
        if type(other) is not RatioVariable:
            return False
        
        return self._num == other._num and self._denom == other._denom

    def set_collection_name(self, collection_name):
        self._num.set_collection_name(collection_name)
        self._denom.set_collection_name(collection_name)

class ProductVariable(VariableBase):
    def __init__(self, var1 : VariableProtocol | str, var2 : VariableProtocol | str):
        self._var1 = BasicVariable(var1) if isinstance(var1, str) else var1
        self._var2 = BasicVariable(var2) if isinstance(var2, str) else var2

    @property
    def _natural_centerline(self):
        return None
    
    @property
    def prebinned(self) -> bool:
        return False
    
    @property
    def columns(self):
        return self._var1.columns + self._var2.columns

    def evaluate(self, dataset, cut):
        return self._var1.evaluate(dataset, cut) * self._var2.evaluate(dataset, cut)

    @property
    def key(self):
        return "%s_times_%s"%(self._var1.key, self._var2.key)
    
    def __eq__(self, other):
        if type(other) is not ProductVariable:
            return False
        
        return self._var1 == other._var1 and self._var2 == other._var2

    def set_collection_name(self, collection_name):
        self._var1.set_collection_name(collection_name)
        self._var2.set_collection_name(collection_name)
        
class DifferenceVariable(VariableBase):
    def __init__(self, var1 : VariableProtocol | str, var2 : VariableProtocol | str):
        self._var1 = BasicVariable(var1) if isinstance(var1, str) else var1
        self._var2 = BasicVariable(var2) if isinstance(var2, str) else var2

    @property
    def _natural_centerline(self):
        return 0.0
    
    @property
    def prebinned(self) -> bool:
        return False
    
    @property
    def columns(self):
        return list(set(self._var1.columns + self._var2.columns))

    def evaluate(self, dataset, cut):
        return self._var2.evaluate(dataset, cut) - self._var1.evaluate(dataset, cut)

    @property
    def key(self):
        return "%s_minus_%s"%(self._var2.key, self._var1.key)
    
    def __eq__(self, other):
        if type(other) is not DifferenceVariable:
            return False
        
        return self._var1 == other._var1 and self._var2 == other._var2

    def set_collection_name(self, collection_name):
        self._var1.set_collection_name(collection_name)
        self._var2.set_collection_name(collection_name)

class SumVariable(VariableBase):
    def __init__(self, var1 : VariableProtocol | str, var2 : VariableProtocol | str):
        self._var1 = BasicVariable(var1) if isinstance(var1, str) else var1
        self._var2 = BasicVariable(var2) if isinstance(var2, str) else var2

    @property
    def _natural_centerline(self):
        return None
    
    @property
    def prebinned(self) -> bool:
        return False
    
    @property
    def columns(self):
        return list(set(self._var1.columns + self._var2.columns))

    def evaluate(self, dataset, cut):
        return self._var1.evaluate(dataset, cut) + self._var2.evaluate(dataset, cut)

    @property
    def key(self):
        return "%s_plus_%s"%(self._var1.key, self._var2.key)
    
    def __eq__(self, other):
        if type(other) is not SumVariable:
            return False
        return self._var1 == other._var1 and self._var2 == other._var2

    def set_collection_name(self, collection_name):
        self._var1.set_collection_name(collection_name)
        self._var2.set_collection_name(collection_name)

class CorrectionlibVariable(VariableBase):
    def __init__(self, var_l : Sequence[VariableProtocol | str], path : str, key : str):
        self._vars = [BasicVariable(var) if isinstance(var, str) else var for var in var_l]

        from correctionlib import CorrectionSet
        cset = CorrectionSet.from_file(path)
        if key not in list(cset.keys()):
            print("Error: Correctionlib key '%s' not found in %s"%(key, path))
            print("Available keys: %s"%list(cset.keys()))
            raise ValueError("Correctionlib key not found")
        self._eval = cset[key].evaluate
        self._csetkey = key

    @property
    def _natural_centerline(self):
        return None
    
    @property
    def prebinned(self) -> bool:
        return False
    
    @property
    def columns(self):
        cols = []
        for var in self._vars:
            cols += var.columns
        return list(set(cols))

    def evaluate(self, dataset, cut):
        args = []
        for var in self._vars:
            args.append(var.evaluate(dataset, cut))

        return self._eval(*args)

    @property
    def key(self):
        return "CORRECTIONLIB(%s)"%(self._csetkey) 
    
    def set_collection_name(self, collection_name):
        for var in self._vars:
            var.set_collection_name(collection_name)

    def __eq__(self, other):
        if type(other) is not CorrectionlibVariable:
            return False
        
        if self._csetkey != other._csetkey:
            return False

        if len(self._vars) != len(other._vars):
            return False

        for i in range(len(self._vars)):
            if self._vars[i] != other._vars[i]:
                return False

        return True
    
class UFuncVariable(VariableBase):
    def __init__(self, var : VariableProtocol | str, ufunc):
        self._var = BasicVariable(var) if isinstance(var, str) else var
        self._ufunc = ufunc

    @property
    def _natural_centerline(self):
        return None
    
    @property
    def prebinned(self) -> bool:
        return False
    
    @property
    def columns(self):
        return self._var.columns

    def evaluate(self, dataset, cut):
        return self._ufunc(self._var.evaluate(dataset, cut))

    @property
    def key(self):
        return 'UFUNC%s(%s)'%(self._ufunc.__name__.replace('<','(').replace('>',')'), self._var.key)
    
    def __eq__(self, other):
        if type(other) is not UFuncVariable:
            return False
        
        return self._var == other._var and self._ufunc == other._ufunc

    def set_collection_name(self, collection_name):
        self._var.set_collection_name(collection_name)

'''
class RateVariable(VariableBase):
    def __init__(self, binaryfield, wrt):
        if type(binaryfield) is str:
            self.binaryfield = BasicVariable(binaryfield)
        else:
            self.binaryfield = binaryfield

        if type(wrt) is str:
            self.wrt = BasicVariable(wrt)
        else:
            self.wrt = wrt

    @property
    def _natural_centerline(self):
        return None
    
    @property
    def prebinned(self) -> bool:
        return False
    
    @property
    def columns(self):
        return list(set(self.binaryfield.columns + self.wrt.columns))

    def evaluate(self, dataset):
        return [self.binaryfield.evaluate(dataset),
                self.wrt.evaluate(dataset)]

    @property
    def key(self):
        return "%s_rate_wrt_%s"%(self.binaryfield.key, self.wrt.key)

    def __eq__(self, other):
        if type(other) is not RateVariable:
            return False
        return self.binaryfield == other.binaryfield and self.wrt == other.wrt

    def set_collection_name(self, collection_name):
        self.binaryfield.set_collection_name(collection_name)
        self.wrt.set_collection_name(collection_name)
'''

class ConcatVariable(VariableBase):
    def __init__(self, vars : Sequence[VariableProtocol | str], keyvar : VariableProtocol | str | None = None):
        self._vars = [BasicVariable(var) if isinstance(var, str) else var for var in vars]
        if keyvar is None:
            print("WARNING: ConcatVariable without key! Automatic labels will fail")
            self._keyvar = BasicVariable("NoKey")
        elif isinstance(keyvar, str):
            self._keyvar = BasicVariable(keyvar)
        else:
            self._keyvar = keyvar

    @property
    def _natural_centerline(self):
        return self._keyvar.centerline

    @property
    def prebinned(self) -> bool:
        return False
    
    @staticmethod
    def build_for_collections(var : VariableProtocol | str, collections_l : List[str]):
        if isinstance(var, str):
            var = BasicVariable(var)
        vars = []
        for coll in collections_l:
            v = copy.deepcopy(var)
            v.set_collection_name(coll)
            vars.append(v)

        result = ConcatVariable(vars, keyvar=var)
        return result

    @property
    def columns(self):
        cols = []
        for var in self._vars:
            cols += var.columns
        return list(set(cols))
    
    def evaluate(self, dataset, cut):
        arrays = []
        for var in self._vars:
            arrays.append(var.evaluate(dataset, cut))
        
        return ak.concatenate(arrays)
    
    @property
    def key(self):
        return self._keyvar.key
    
    def __eq__(self, other):
        if type(other) is not ConcatVariable:
            return False

        return self._keyvar == other._keyvar        

    def set_collection_name(self, collection_name):
        print("WARNING: overwriting collection name for all variables in ConcatVariable object")
        
        for var in self._vars:
            var.set_collection_name(collection_name)

        self._keyvar.set_collection_name(collection_name)
