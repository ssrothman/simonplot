import copy

from simon_mpl_util.plotting.util.config import lookup_axis_label

from typing import List

def variable_from_string(name):
    if 'over' in name:
        num, denom = name.split('_over_')
        return RatioVariable(num, denom)
    elif 'times' in name:
        var1, var2 = name.split('_times_')
        return ProductVariable(var1, var2)
    else:
        return BasicVariable(name)

class AbstractVariable:
    @property
    def columns(self):
        raise NotImplementedError()

    def evaluate(self, dataset):
        raise NotImplementedError()

    @property
    def key(self):
        raise NotImplementedError()
    
    def override_label(self, label):
        self._label = label

    def clear_override_label(self):
        if hasattr(self, '_label'):
            del self._label

    @property
    def label(self):
        if hasattr(self, '_label') and self._label is not None:
            return self._label
        else:
            return lookup_axis_label(self.key)

    def __eq__(self, other):
        raise NotImplementedError()

    def set_collection_name(self, collection_name):
        raise NotImplementedError()

class PrebinnedVariable(AbstractVariable):
    @property 
    def columns(self):
        return []
    
    def evaluate(self, dataset):
        raise RuntimeError("PrebinnedVariable objects are just placeholders and cannot be evaluated. The actual evaluation happens in the cut object.")
    
    @property 
    def key(self):
        return "PREBINNED"
    
    def __eq__(self, other):
        return isinstance(other, PrebinnedVariable)
    
    def set_collection_name(self, collection_name):
        pass

class ConstantVariable(AbstractVariable):
    def __init__(self, value):
        self._value = value

    @property
    def columns(self):
        return []
    
    def evaluate(self, dataset):
        return self._value
    
    @property
    def key(self):
        return "CONST(%s)"%(self._value)
    
    def __eq__(self, other):
        if type(other) is not ConstantVariable:
            return False
        
        return self._value == other._value
    
    def set_collection_name(self, collection_name):
        pass

class BasicVariable(AbstractVariable):
    def __init__(self, name, collection_name=None):
        self._name = name
        self._collection_name = collection_name

        if "." in name and collection_name is None:
            splitted = name.split(".")
            self._collection_name = ".".join(splitted[:-1])
            self._name = splitted[-1]
        elif '.' in name:
            raise ValueError("Variable name '%s' contains '.' but collection_name is also specified"%(name))

    @property
    def columns(self):
        if self._collection_name is None:
            return [self._name]
        else:
            return [self._collection_name + "." + self._name]

    def evaluate(self, dataset):
        return dataset.get_column(self._name, self._collection_name)
    
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

class ConcatVariable(AbstractVariable):
    def __init__(self, *vars, keyvar=None):
        self.vars_ = vars
        if keyvar is None:
            print("WARNING: ConcatVariable without key! Automatic labels will fail")
            self._keyvar = BasicVariable("None")
        else:
            self._keyvar = keyvar

    @staticmethod
    def build_for_collections(var : AbstractVariable, collections_l : List[str]):
        vars = []
        for coll in collections_l:
            v = copy.deepcopy(var)
            v.set_collection_name(coll)
            vars.append(v)

        result = ConcatVariable(*vars, keyvar=var)
        return result

    @property
    def columns(self):
        cols = []
        for var in self.vars_:
            cols += var.columns
        return list(set(cols))
    
    def evaluate(self, dataset):
        import awkward as ak

        arrays = []
        for var in self.vars_:
            arrays.append(var.evaluate(dataset))
        
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
        for var in self.vars_:
            var.set_collection_name(collection_name)
        self._keyvar.set_collection_name(collection_name)


class AkNumVariable(AbstractVariable):
    def __init__(self, name):
        self.name = name

    @property
    def columns(self):
        return [self.name]

    def evaluate(self, dataset):
        return dataset.get_aknum_column(self.name)
    
    @property
    def key(self):
        return "N(%s)"%(self.name)
    
    def __eq__(self, other):
        if type(other) is not AkNumVariable:
            return False
        
        return self.name == other.name
    
    def set_collection_name(self, collection_name):
        raise ValueError("AkNumVariable does not support set_collection_name")

class RatioVariable(AbstractVariable):
    def __init__(self, num, denom):
        if type(num) is str:
            self.num = BasicVariable(num)
        else:
            self.num = num

        if type(denom) is str:
            self.denom = BasicVariable(denom)
        else:
            self.denom = denom

    @property
    def columns(self):
        return list(set(self.num.columns + self.denom.columns))

    def evaluate(self, dataset):
        return self.num.evaluate(dataset) / self.denom.evaluate(dataset)

    @property
    def key(self):
        return "%s_over_%s"%(self.num.key, self.denom.key)
    
    def __eq__(self, other):
        if type(other) is not RatioVariable:
            return False
        
        return self.num == other.num and self.denom == other.denom

    def set_collection_name(self, collection_name):
        self.num.set_collection_name(collection_name)
        self.denom.set_collection_name(collection_name)

class ProductVariable(AbstractVariable):
    def __init__(self, var1, var2):
        if type(var1) is str:
            self.var1 = BasicVariable(var1)
        else:
            self.var1 = var1

        if type(var2) is str:
            self.var2 = BasicVariable(var2)
        else:
            self.var2 = var2

    @property
    def columns(self):
        return self.var1.columns + self.var2.columns

    def evaluate(self, dataset):
        return self.var1.evaluate(dataset) * self.var2.evaluate(dataset)

    @property
    def key(self):
        return "%s_times_%s"%(self.var1.key, self.var2.key)
    
    def __eq__(self, other):
        if type(other) is not ProductVariable:
            return False
        
        return self.var1 == other.var1 and self.var2 == other.var2

    def set_collection_name(self, collection_name):
        self.var1.set_collection_name(collection_name)
        self.var2.set_collection_name(collection_name)

class DifferenceVariable(AbstractVariable):
    def __init__(self, gen, reco):
        self.gen = gen
        self.reco = reco

        if type(gen) is str:
            self.gen = variable_from_string(gen)
        if type(reco) is str:
            self.reco = variable_from_string(reco)

    @property
    def columns(self):
        return list(set(self.gen.columns + self.reco.columns))

    def evaluate(self, dataset):
        return self.reco.evaluate(dataset) - self.gen.evaluate(dataset)

    @property
    def key(self):
        return "%s_minus_%s"%(self.reco.key, self.gen.key)
    
    def __eq__(self, other):
        if type(other) is not DifferenceVariable:
            return False
        return self.gen == other.gen and self.reco == other.reco

    def set_collection_name(self, collection_name):
        self.gen.set_collection_name(collection_name)
        self.reco.set_collection_name(collection_name)

class SumVariable(AbstractVariable):
    def __init__(self, x1, x2):
        self.x1 = x1
        self.x2 = x2

        if type(x1) is str:
            self.x1 = variable_from_string(x1)
        if type(x2) is str:
            self.x2 = variable_from_string(x2)

    @property
    def columns(self):
        return list(set(self.x1.columns + self.x2.columns))

    def evaluate(self, dataset):
        return self.x1.evaluate(dataset) + self.x2.evaluate(dataset)

    @property
    def key(self):
        return "%s_plus_%s"%(self.x1.key, self.x2.key)
    
    def __eq__(self, other):
        if type(other) is not SumVariable:
            return False
        return self.x1 == other.x1 and self.x2 == other.x2

    def set_collection_name(self, collection_name):
        self.x1.set_collection_name(collection_name)
        self.x2.set_collection_name(collection_name)

class CorrectionlibVariable(AbstractVariable):
    def __init__(self, var_l, path, key):
        self.var_l = []
        for var in var_l:
            if type(var) is str:
                self.var_l.append(variable_from_string(var))
            else:
                self.var_l.append(var)

        from correctionlib import CorrectionSet
        cset = CorrectionSet.from_file(path)
        if key not in list(cset.keys()):
            print("Error: Correctionlib key '%s' not found in %s"%(key, path))
            print("Available keys: %s"%list(cset.keys()))
            raise ValueError("Correctionlib key not found")
        self.eval = cset[key].evaluate
        self.csetkey = key

    @property
    def columns(self):
        cols = []
        for var in self.var_l:
            cols += var.columns
        return list(set(cols))

    def evaluate(self, dataset):
        args = []
        for var in self.var_l:
            args.append(var.evaluate(dataset))

        return self.eval(*args)

    @property
    def key(self):
        return "CORRECTIONLIB(%s)"%(self.csetkey)

    def set_collection_name(self, collection_name):
        for var in self.var_l:
            var.set_collection_name(collection_name)
    
class UFuncVariable(AbstractVariable):
    def __init__(self, var, ufunc):
        if type(var) is str:
            self.var = variable_from_string(var)
        else:
            self.var = var

        self.ufunc = ufunc

    @property
    def columns(self):
        return self.var.columns

    def evaluate(self, dataset):
        return self.ufunc(self.var.evaluate(dataset))

    @property
    def key(self):
        return 'UFUNC%s(%s)'%(self.ufunc.__name__, self.var.key)
    
    def __eq__(self, other):
        if type(other) is not UFuncVariable:
            return False
        return self.var == other.var and self.ufunc == other.ufunc

    def set_collection_name(self, collection_name):
        self.var.set_collection_name(collection_name)

class RateVariable(AbstractVariable):
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