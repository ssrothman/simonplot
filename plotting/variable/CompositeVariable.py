from .Variable import AbstractVariable, variable_from_string, UFuncVariable, SumVariable, DifferenceVariable

from simon_mpl_util.util.coordinates import xyz_to_eta_phi

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
        return list(set(self.gen.columns + self.reco.columns))
    
    def evaluate(self, dataset):
        gen = self.gen.evaluate(dataset)
        reco = self.reco.evaluate(dataset)
        return (reco - gen) / gen

    @property
    def key(self):
        return "%s_minus_%s_over_%s"%(self.reco.key, self.gen.key, self.gen.key)
    
    def __eq__(self, other):
        if type(other) is not RelativeResolutionVariable:
            return False
        return self.gen == other.gen and self.reco == other.reco

    def set_collection_name(self, collection_name):
        self.gen.set_collection_name(collection_name)
        self.reco.set_collection_name(collection_name)

class Magnitude3dVariable(AbstractVariable):
    def __init__(self, xvar, yvar, zvar):
        import numpy as np

        self.xvar = xvar
        self.yvar = yvar
        self.zvar = zvar

        self.x2var = UFuncVariable(self.xvar, np.square)
        self.y2var = UFuncVariable(self.yvar, np.square)
        self.z2var = UFuncVariable(self.zvar, np.square)

        self.r2var = SumVariable(
            SumVariable(self.x2var, self.y2var),
            self.z2var
        )

        self.rvar = UFuncVariable(self.r2var, np.sqrt)
    
    @property
    def columns(self):
        return list(set(
            self.xvar.columns +
            self.yvar.columns +
            self.zvar.columns
        ))  
    
    def evaluate(self, dataset):
        return self.rvar.evaluate(dataset)
    
    @property
    def key(self):
        return "sqrt(%s^2 + %s^2 + %s^2)"%(self.xvar.key, self.yvar.key, self.zvar.key)

    def __eq__(self, other):
        if type(other) is not Magnitude3dVariable:
            return False
        return (self.xvar == other.xvar and
                self.yvar == other.yvar and
                self.zvar == other.zvar)

    def set_collection_name(self, collection_name):
        self.rvar.set_collection_name(collection_name)
        self.xvar.set_collection_name(collection_name)
        self.yvar.set_collection_name(collection_name)
        self.zvar.set_collection_name(collection_name)

class Magnitude2dVariable(AbstractVariable):
    def __init__(self, xvar, yvar):
        import numpy as np

        self.xvar = xvar
        self.yvar = yvar

        self.x2var = UFuncVariable(self.xvar, np.square)
        self.y2var = UFuncVariable(self.yvar, np.square)

        self.r2var = SumVariable(self.x2var, self.y2var)
        self.rvar = UFuncVariable(self.r2var, np.sqrt)
    
    @property
    def columns(self):
        return list(set(
            self.xvar.columns +
            self.yvar.columns
        ))  
    
    def evaluate(self, dataset):
        return self.rvar.evaluate(dataset)
    
    @property
    def key(self):
        return "sqrt(%s^2 + %s^2)"%(self.xvar.key, self.yvar.key)

    def __eq__(self, other):
        if type(other) is not Magnitude3dVariable:
            return False
        return (self.xvar == other.xvar and
                self.yvar == other.yvar)

    def set_collection_name(self, collection_name):
        self.rvar.set_collection_name(collection_name)
        self.xvar.set_collection_name(collection_name)
        self.yvar.set_collection_name(collection_name)

class Distance3dVariable(AbstractVariable):
    def __init__(self, x1var, y1var, z1var, x2var, y2var, z2var):
        import numpy as np

        self.dxvar = DifferenceVariable(x1var, x2var)
        self.dyvar = DifferenceVariable(y1var, y2var)
        self.dzvar = DifferenceVariable(z1var, z2var)

        self.magnitude_var = Magnitude3dVariable(
            self.dxvar,
            self.dyvar,
            self.dzvar
        )

    @property
    def columns(self):
        return list(set(
            self.dxvar.columns +
            self.dyvar.columns +
            self.dzvar.columns
        ))  
    
    def evaluate(self, dataset):
        return self.magnitude_var.evaluate(dataset)
    
    @property
    def key(self):
        return "Distance3D(%s_%s_%s - %s_%s_%s)"%(
            self.dxvar.gen.key,
            self.dyvar.gen.key,
            self.dzvar.gen.key,
            self.dxvar.reco.key,
            self.dyvar.reco.key,
            self.dzvar.reco.key
        )
    
    def __eq__(self, other):
        if type(other) is not Distance3dVariable:
            return False
        return (self.dxvar == other.dxvar and
                self.dyvar == other.dyvar and
                self.dzvar == other.dzvar)
    
    def set_collection_name(self, collection_name):
        self.dxvar.set_collection_name(collection_name)
        self.dyvar.set_collection_name(collection_name)
        self.dzvar.set_collection_name(collection_name)
        self.magnitude_var.set_collection_name(collection_name)

class EtaFromXYZVariable(AbstractVariable):
    def __init__(self, x, y, z):
        self._x = x
        self._y = y
        self._z = z

    @property 
    def columns(self):
        return list(set(self._x.columns + self._y.columns + self._z.columns))
    
    @property
    def key(self):
        return "ETA(%s_%s_%s)" % (self._x.key, self._y.key, self._z.key)
    
    def __eq__(self, other):
        if type(other) is not EtaFromXYZVariable:
            return False
        
        return (self._x == other._x and 
                self._y == other._y and
                self._z == other._z)
    
    def set_collection_name(self, collection_name):
        self._x.set_collection_name(collection_name)
        self._y.set_collection_name(collection_name)
        self._z.set_collection_name(collection_name)

    def evaluate(self, dataset):
        xval = self._x.evaluate(dataset)
        yval = self._y.evaluate(dataset)
        zval = self._z.evaluate(dataset)

        return xyz_to_eta_phi(xval, yval, zval)[0]
    
class PhiFromXYZVariable(AbstractVariable):
    def __init__(self, x, y, z):
        self._x = x
        self._y = y
        self._z = z

    @property 
    def columns(self):
        return list(set(self._x.columns + self._y.columns + self._z.columns))
    
    @property
    def key(self):
        return "PHI(%s_%s_%s)" % (self._x.key, self._y.key, self._z.key)
    
    def __eq__(self, other):
        if type(other) is not EtaFromXYZVariable:
            return False
        
        return (self._x == other._x and 
                self._y == other._y and
                self._z == other._z)
    
    def set_collection_name(self, collection_name):
        self._x.set_collection_name(collection_name)
        self._y.set_collection_name(collection_name)
        self._z.set_collection_name(collection_name)

    def evaluate(self, dataset):
        xval = self._x.evaluate(dataset)
        yval = self._y.evaluate(dataset)
        zval = self._z.evaluate(dataset)

        return xyz_to_eta_phi(xval, yval, zval)[1]