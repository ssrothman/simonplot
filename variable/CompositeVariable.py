from .Variable import BasicVariable, UFuncVariable, SumVariable, DifferenceVariable
from .VariableBase import VariableBase
from simonplot.typing.Protocols import VariableProtocol

from simonpy.coordinates import xyz_to_eta_phi

class RelativeResolutionVariable(VariableBase):
    def __init__(self, gen : VariableProtocol | str, reco : VariableProtocol | str):
        if isinstance(gen, str):
            gen = BasicVariable(gen)
        if isinstance(reco, str):
            reco = BasicVariable(reco)

        self._gen = gen
        self._reco = reco

    @property
    def _natural_centerline(self):
        return 0.0
    
    @property
    def prebinned(self) -> bool:
        return False

    @property
    def columns(self):
        return list(set(self._gen.columns + self._reco.columns))
    
    def evaluate(self, dataset, cut):
        gen = self._gen.evaluate(dataset, cut)
        reco = self._reco.evaluate(dataset, cut)
        return (reco - gen) / gen

    @property
    def key(self):
        return "%s_minus_%s_over_%s"%(self._reco.key, self._gen.key, self._gen.key)
    
    def __eq__(self, other):
        if type(other) is not RelativeResolutionVariable:
            return False
        return self._gen == other._gen and self._reco == other._reco

    def set_collection_name(self, collection_name):
        self._gen.set_collection_name(collection_name)
        self._reco.set_collection_name(collection_name)

class Magnitude3dVariable(VariableBase):
    def __init__(self, xvar: VariableProtocol | str, yvar: VariableProtocol | str, zvar: VariableProtocol | str):
        import numpy as np

        if isinstance(xvar, str):
            xvar = BasicVariable(xvar)
        if isinstance(yvar, str):
            yvar = BasicVariable(yvar)
        if isinstance(zvar, str):
            zvar = BasicVariable(zvar)

        self._xvar = xvar
        self._yvar = yvar
        self._zvar = zvar

        self._x2var = UFuncVariable(self._xvar, np.square)
        self._y2var = UFuncVariable(self._yvar, np.square)
        self._z2var = UFuncVariable(self._zvar, np.square)

        self.r2var = SumVariable(
            SumVariable(self._x2var, self._y2var),
            self._z2var
        )

        self._rvar = UFuncVariable(self.r2var, np.sqrt)
    
    @property
    def _natural_centerline(self):
        return None
    
    @property
    def prebinned(self) -> bool:
        return False

    @property
    def columns(self):
        return list(set(
            self._xvar.columns +
            self._yvar.columns +
            self._zvar.columns
        ))  
    
    def evaluate(self, dataset, cut):
        return self._rvar.evaluate(dataset, cut)
    
    @property
    def key(self):
        return "sqrt(%s^2 + %s^2 + %s^2)"%(self._xvar.key, self._yvar.key, self._zvar.key)

    def __eq__(self, other):
        if type(other) is not Magnitude3dVariable:
            return False
        return (self._xvar == other._xvar and
                self._yvar == other._yvar and
                self._zvar == other._zvar)

    def set_collection_name(self, collection_name):
        self._rvar.set_collection_name(collection_name)
        self._xvar.set_collection_name(collection_name)
        self._yvar.set_collection_name(collection_name)
        self._zvar.set_collection_name(collection_name)

class Magnitude2dVariable(VariableBase):
    def __init__(self, xvar: VariableProtocol | str, yvar: VariableProtocol | str):
        import numpy as np

        if isinstance(xvar, str):
            xvar = BasicVariable(xvar)
        if isinstance(yvar, str):
            yvar = BasicVariable(yvar)

        self._xvar = xvar
        self._yvar = yvar

        self._x2var = UFuncVariable(self._xvar, np.square)
        self._y2var = UFuncVariable(self._yvar, np.square)

        self._r2var = SumVariable(self._x2var, self._y2var)
        self._rvar = UFuncVariable(self._r2var, np.sqrt)
    
    @property
    def _natural_centerline(self):
        return None
    
    @property
    def prebinned(self) -> bool:
        return False
    
    @property
    def columns(self):
        return list(set(
            self._xvar.columns +
            self._yvar.columns
        ))  
    
    def evaluate(self, dataset, cut):
        return self._rvar.evaluate(dataset, cut)
    
    @property
    def key(self):
        return "sqrt(%s^2 + %s^2)"%(self._xvar.key, self._yvar.key)

    def __eq__(self, other):
        if type(other) is not Magnitude3dVariable:
            return False
        return (self._xvar == other._xvar and
                self._yvar == other._yvar)

    def set_collection_name(self, collection_name):
        self._rvar.set_collection_name(collection_name)
        self._xvar.set_collection_name(collection_name)
        self._yvar.set_collection_name(collection_name)

class Distance3dVariable(VariableBase):
    def __init__(self, x1var: VariableProtocol | str, y1var: VariableProtocol | str, z1var: VariableProtocol | str, x2var: VariableProtocol | str, y2var: VariableProtocol | str, z2var: VariableProtocol | str):
        import numpy as np

        if isinstance(x1var, str):
            x1var = BasicVariable(x1var)
        if isinstance(y1var, str):
            y1var = BasicVariable(y1var)
        if isinstance(z1var, str):
            z1var = BasicVariable(z1var)
        if isinstance(x2var, str):
            x2var = BasicVariable(x2var)
        if isinstance(y2var, str):
            y2var = BasicVariable(y2var)
        if isinstance(z2var, str):
            z2var = BasicVariable(z2var)

        self._dxvar = DifferenceVariable(x1var, x2var)
        self._dyvar = DifferenceVariable(y1var, y2var)
        self._dzvar = DifferenceVariable(z1var, z2var)

        self.magnitude_var = Magnitude3dVariable(
            self._dxvar,
            self._dyvar,
            self._dzvar
        )

    @property
    def _natural_centerline(self):
        return None
    
    @property
    def prebinned(self) -> bool:
        return False
    
    @property
    def columns(self):
        return list(set(
            self._dxvar.columns +
            self._dyvar.columns +
            self._dzvar.columns
        ))  
    
    def evaluate(self, dataset, cut):
        return self.magnitude_var.evaluate(dataset, cut)
    
    @property
    def key(self):
        return "Distance3D(%s_%s_%s - %s_%s_%s)"%(
            self._dxvar._var2.key,
            self._dyvar._var2.key,
            self._dzvar._var2.key,
            self._dxvar._var1.key,
            self._dyvar._var1.key,
            self._dzvar._var1.key
        )
    
    def __eq__(self, other):
        if type(other) is not Distance3dVariable:
            return False
        return (self._dxvar == other._dxvar and
                self._dyvar == other._dyvar and
                self._dzvar == other._dzvar)
    
    def set_collection_name(self, collection_name):
        self._dxvar.set_collection_name(collection_name)
        self._dyvar.set_collection_name(collection_name)
        self._dzvar.set_collection_name(collection_name)
        self.magnitude_var.set_collection_name(collection_name)

class EtaFromXYZVariable(VariableBase):
    def __init__(self, x : VariableProtocol | str, y: VariableProtocol | str, z: VariableProtocol | str):
        if isinstance(x, str):
            x = BasicVariable(x)
        if isinstance(y, str):
            y = BasicVariable(y)
        if isinstance(z, str):
            z = BasicVariable(z)

        self._x = x
        self._y = y
        self._z = z

    @property
    def _natural_centerline(self):
        return None
    
    @property
    def prebinned(self) -> bool:
        return False
    
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

    def evaluate(self, dataset, cut):
        xval = self._x.evaluate(dataset, cut)
        yval = self._y.evaluate(dataset, cut)
        zval = self._z.evaluate(dataset, cut)

        return xyz_to_eta_phi(xval, yval, zval)[0]
    
class PhiFromXYZVariable(VariableBase):
    def __init__(self, x : VariableProtocol | str, y: VariableProtocol | str, z: VariableProtocol | str):
        if isinstance(x, str):
            x = BasicVariable(x)
        if isinstance(y, str):
            y = BasicVariable(y)
        if isinstance(z, str):
            z = BasicVariable(z)

        self._x = x
        self._y = y
        self._z = z

    @property
    def _natural_centerline(self):
        return None
    
    @property
    def prebinned(self) -> bool:
        return False
    
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

    def evaluate(self, dataset, cut):
        xval = self._x.evaluate(dataset, cut)
        yval = self._y.evaluate(dataset, cut)
        zval = self._z.evaluate(dataset, cut)

        return xyz_to_eta_phi(xval, yval, zval)[1]