from .SetupConfig import config, lookup_axis_label

def variable_from_string(name):
    if 'over' in name:
        num, denom = name.split('_over_')
        return RatioVariable(num, denom)
    elif 'times' in name:
        var1, var2 = name.split('_times_')
        return ProductVariable(var1, var2)
    else:
        return Variable(name)

class AbstractVariable:
    @property
    def columns(self):
        raise NotImplementedError()

    def evaluate(self, table):
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

class Variable(AbstractVariable):
    def __init__(self, name):
        self.name = name

    @property
    def columns(self):
        return [self.name]

    def evaluate(self, table):
        return table.get_column(self.name)
    
    @property
    def key(self):
        return self.name
    
class AkNumVariable(AbstractVariable):
    def __init__(self, name):
        self.name = name

    @property
    def columns(self):
        return [self.name]

    def evaluate(self, table):
        return table.get_aknum_column(self.name)
    
    @property
    def key(self):
        return "N(%s)"%(self.name)

class RatioVariable(AbstractVariable):
    def __init__(self, num, denom):
        if type(num) is str:
            self.num = Variable(num)
        else:
            self.num = num

        if type(denom) is str:
            self.denom = Variable(denom)
        else:
            self.denom = denom

    @property
    def columns(self):
        return list(set(self.num.columns + self.denom.columns))

    def evaluate(self, table):
        return self.num.evaluate(table) / self.denom.evaluate(table)

    @property
    def key(self):
        return "%s_over_%s"%(self.num.key, self.denom.key)

class ProductVariable(AbstractVariable):
    def __init__(self, var1, var2):
        if type(var1) is str:
            self.var1 = Variable(var1)
        else:
            self.var1 = var1

        if type(var2) is str:
            self.var2 = Variable(var2)
        else:
            self.var2 = var2

    @property
    def columns(self):
        return self.var1.columns + self.var2.columns

    def evaluate(self, table):
        return self.var1.evaluate(table) * self.var2.evaluate(table)

    @property
    def key(self):
        return "%s_times_%s"%(self.var1.key, self.var2.key)
    
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
        return self.gen.columns + self.reco.columns

    def evaluate(self, table):
        return self.reco.evaluate(table) - self.gen.evaluate(table)

    @property
    def key(self):
        return "%s_minus_%s"%(self.reco.key, self.gen.key)
    

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

    def evaluate(self, table):
        args = []
        for var in self.var_l:
            args.append(var.evaluate(table))

        return self.eval(*args)

    @property
    def key(self):
        return "CORRECTIONLIB(%s)"%(self.csetkey)

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

    def evaluate(self, table):
        return self.ufunc(self.var.evaluate(table))

    @property
    def key(self):
        return 'UFUNC%s(%s)'%(self.ufunc.__name__, self.var.key)
    
class RateVariable(AbstractVariable):
    def __init__(self, binaryfield, wrt):
        if type(binaryfield) is str:
            self.binaryfield = Variable(binaryfield)
        else:
            self.binaryfield = binaryfield

        if type(wrt) is str:
            self.wrt = Variable(wrt)
        else:
            self.wrt = wrt

    @property
    def columns(self):
        return self.binaryfield.columns + self.wrt.columns

    def evaluate(self, table):
        return [self.binaryfield.evaluate(table),
                self.wrt.evaluate(table)]

    @property
    def key(self):
        return "%s_rate_wrt_%s"%(self.binaryfield.key, self.wrt.key)

class RelativeResolutionVariable(AbstractVariable):
    def __init__(self, gen, reco):
        self.gen = gen
        self.reco = reco

        if type(gen) is str:
            self.gen = variable_from_string(gen)
        if type(reco) is str:
            self.reco = variable_from_string(reco)

    @property
    def columns(self):
        return self.gen.columns + self.reco.columns
    
    def evaluate(self, table):
        gen = self.gen.evaluate(table)
        reco = self.reco.evaluate(table)
        return (reco - gen) / gen

    @property
    def key(self):
        return "%s_minus_%s_over_%s"%(self.reco.key, self.gen.key, self.gen.key)