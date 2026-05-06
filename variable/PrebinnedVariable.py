from turtle import up
from simonplot.typing.Protocols import VariableProtocol
from simonplot.typing.Protocols import PrebinnedOperationProtocol, PrebinnedVariableProtocol
from simonpy.AbitraryBinning import ArbitraryGenRecoBinning
from simonpy.sanitization import maybe_valcov_to_definitely_valcov
from simonpy.stats_v2 import apply_jacobian, divide_out_profile, normalize_per_block
from .VariableBase import VariableBase
from typing import Sequence, Tuple, assert_never, override, List
import numpy as np

class BasicPrebinnedVariable(VariableBase):
    def __init__(self):
        pass #stateless
    
    @property
    def _natural_centerline(self):
        return None
    
    @property 
    def columns(self):
        return []
    
    @property
    def prebinned(self) -> bool:
        return True
    
    def evaluate(self, dataset, cut):
        return cut.evaluate(dataset)

    @property 
    def key(self):
        return "PREBINNED"

    def set_collection_name(self, collection_name):
        raise ValueError("Prebinned Variables do not support set_collection_name")

    def __eq__(self, other) -> bool:
        return isinstance(other, BasicPrebinnedVariable)
    
    @property
    def label(self) -> str:
        return "Bin counts"
    
    @property
    def matrixlabel(self) -> str:
        return "Covariance"
    
    @property
    def hasjacobian(self) -> bool:
        return False

    @property
    def normalized_blocks(self) -> List[str]:
        return []
    
    @property
    def normalized_by_err(self) -> bool:
        return False

    @property
    def jac_details(self) -> dict:
        return {
            'wrt' : [],
            'radial_coords' : [],
            'clip_negativeinf' : {},
            'clip_positiveinf' : {}
        }

class WithJacobian(VariableBase):
    def __init__(self, 
                 variable : PrebinnedVariableProtocol, 
                 wrt : Sequence[str],
                 radial_coords : Sequence[str],
                 clip_negativeinf : dict[str, float] = {},
                 clip_positiveinf : dict[str, float] = {}):
        self._var = variable
        self._wrt = wrt
        self._radial_coords = radial_coords
        self._clip_negativeinf = clip_negativeinf
        self._clip_positiveinf = clip_positiveinf

        if self._var.hasjacobian:
            raise ValueError("WithJacobian should be applied only once!")

    @property
    def _natural_centerline(self):
        return None
    
    @property
    def columns(self):
        return []
    
    @property
    def prebinned(self) -> bool:
        return True
    
    def evaluate(self, dataset, cut):
        if not isinstance(cut, PrebinnedOperationProtocol):
            raise ValueError("PrebinnedDensityVariable requires a PrebinnedOperationProtocol cut")
        
        evaluated = self._var.evaluate(dataset, cut)
        hist, cov, thelen, thedtype = maybe_valcov_to_definitely_valcov(evaluated)

        #after type checks we can be confident that
        #hist = np.ndarray with shape (thelen,), or else None
        #cov = np.ndarray with shape (thelen, thelen), or else None

        binning = cut.resulting_binning(dataset.binning)

        return apply_jacobian(hist, cov, binning, self.jac_details) # type: ignore
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, WithJacobian):
            return False
        return (self._var == other._var and
                self._radial_coords == other._radial_coords and
                self._clip_negativeinf == other._clip_negativeinf and
                self._clip_positiveinf == other._clip_positiveinf)

    @property 
    def key(self):
        return "WithJacobian(%s)" % self._var.key

    def set_collection_name(self, collection_name):
        raise ValueError("Prebinned Variables do not support set_collection_name")

    @property
    def hasjacobian(self) -> bool:
        return True

    @property
    def normalized_blocks(self) -> List[str]:
        return self._var.normalized_blocks
    
    @property
    def normalized_by_err(self) -> bool:
        return self._var.normalized_by_err
                   
    @property
    def jac_details(self) -> dict:
        return {
            'wrt' : self._wrt,
            'radial_coords' : self._radial_coords,
            'clip_negativeinf' : self._clip_negativeinf,
            'clip_positiveinf' : self._clip_positiveinf
        }
    
'''
Nearly identical to NormalizePerBlock, but normalizes to mean value instead of to integral
'''
class DivideOutProfile(VariableBase):
    def __init__(self, variable : PrebinnedVariableProtocol, axes : List[str]):
        self._var = variable
        self._axes = axes

        if self._var.normalized_by_err:
            raise ValueError("DivideOutProfile should be applied BEFORE normalization by error (ie CorrelationFromCovariance)!")

        if self._var.normalized_blocks:
            for v in self._var.normalized_blocks:
                if v in axes:
                    print("Warning: Attempting to normalize in blocks of %s twice!"%v)
                    #raise ValueError("Attempting to normalize in blocks of %s twice!"%v)

    @property
    def _natural_centerline(self):
        return 1.0
    
    @property
    def columns(self):
        return []
    
    @property
    def prebinned(self) -> bool:
        return True
    
    def evaluate(self, dataset, cut):
        if not isinstance(cut, PrebinnedOperationProtocol):
            raise ValueError("DivideOutProfile requires a PrebinnedOperationProtocol cut")

        evaluated = self._var.evaluate(dataset, cut)
        hist, cov, _, _ = maybe_valcov_to_definitely_valcov(evaluated)
        if hist is None:
            raise ValueError("Can't DivideOutProfile without histogram values!")
        
        binning = cut.resulting_binning(dataset.binning)

        return divide_out_profile(hist, cov, binning, self._axes)

    def __eq__(self, other) -> bool:
        if not isinstance(other, DivideOutProfile):
            return False
        return (self._var == other._var and
                self._axes == other._axes)
    
    @property 
    def key(self):
        return "DivideOutProfile(%s-%s)" % (self._var.key, '-'.join(self._axes))
    
    def set_collection_name(self, collection_name):
        raise ValueError("Prebinned Variables do not support set_collection_name")
    
    @property
    def hasjacobian(self) -> bool:
        return self._var.hasjacobian
    
    @property
    def normalized_blocks(self) -> List[str]:
        return self._axes + self._var.normalized_blocks
    
    @property
    def normalized_by_err(self) -> bool:
        return self._var.normalized_by_err
    
    @property
    def jac_details(self) -> dict:
        return self._var.jac_details
    
class NormalizePerBlock(VariableBase):
    def __init__(self, variable : PrebinnedVariableProtocol, axes : List[str]):
        self._var = variable
        self._axes = axes
        if self._var.hasjacobian:
            raise ValueError("NormalizePerBlock should be applied BEFORE jacobian!")
        if self._var.normalized_by_err:
            raise ValueError("NormalizePerBlock should be applied BEFORE normalization by error (ie CorrelationFromCovariance)!")
        if self._var.normalized_blocks:
            for v in self._var.normalized_blocks:
                if v in axes:
                    print("Warning: Attempting to normalize in blocks of %s twice!"%v)
        
    @property
    def _natural_centerline(self):
        return None
    
    @property
    def columns(self):
        return []
    
    @property
    def prebinned(self) -> bool:
        return True
    
    def evaluate(self, dataset, cut):
        if not isinstance(cut, PrebinnedOperationProtocol):
            raise ValueError("PrebinnedDensityVariable requires a PrebinnedOperationProtocol cut")
        
        evaluated = self._var.evaluate(dataset, cut)
        hist, cov, _, _ = maybe_valcov_to_definitely_valcov(evaluated)
        if hist is None:
            raise ValueError("Can't NormalizerPerBlock without histogram values!")
        
        print("Normalize per block:")
        print("\tvariable: %s"%self._var.key)
        print("\tblocks: %s"%self._axes)
        print("\tdataset: %s (%s)"%(dataset.key, type(dataset)))
        print("\tcut: %s (%s)"%(cut.key, type(cut)))

        binning = cut.resulting_binning(dataset.binning)
        print("\tbinning: %s"%binning)

        return normalize_per_block(
            hist, cov, binning, self._axes
        )
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, NormalizePerBlock):
            return False
        return (self._var == other._var and
                self._axes == other._axes)
    
    @property 
    def key(self):
        return "NormalizePerBlock(%s-%s)" % (self._var.key, '-'.join(self._axes))

    def set_collection_name(self, collection_name):
        raise ValueError("Prebinned Variables do not support set_collection_name")

    @property
    def covlabel(self) -> str:
        return "Covariance on normalized blocks"

    @property
    def hasjacobian(self) -> bool:
        return self._var.hasjacobian

    @property
    def normalized_blocks(self) -> List[str]:
        return self._axes + self._var.normalized_blocks
    
    @property
    def normalized_by_err(self) -> bool:
        return self._var.normalized_by_err

    @property
    def jac_details(self) -> dict:
        return self._var.jac_details        

class CorrelationFromCovariance(VariableBase):
    def __init__(self, variable : PrebinnedVariableProtocol):
        self._var = variable

        if self._var.normalized_by_err:
            raise ValueError("Attempting to normalizebyerr twice!")

    @property
    def _natural_centerline(self):
        return 0

    @property
    def columns(self):
        return []
    
    @property
    def prebinned(self) -> bool:
        return True
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, CorrelationFromCovariance):
            return False
        return (self._var == other._var)
    
    def evaluate(self, dataset, cut):
        if not isinstance(cut, PrebinnedOperationProtocol):
            raise ValueError("PrebinnedDensityVariable requires a PrebinnedOperationProtocol cut")

        evaluated = self._var.evaluate(dataset, cut)
        hist, cov, _, _ = maybe_valcov_to_definitely_valcov(evaluated)

        if cov is None:
            raise RuntimeError("CorrelationFromCovariance needs covariance!!")
        
        errs = np.sqrt(np.diag(cov))
        outer = np.outer(errs, errs)
        outer[outer==0] = 1 #avoid divide by zero

        correl = cov/outer

        if hist is not None:
            vals = hist/errs 
            return vals, correl
        else:
            return correl

    @property 
    def key(self):
        return "CorrelationFromCovariance(%s)" % self._var.key

    def set_collection_name(self, collection_name):
        raise ValueError("Prebinned Variables do not support set_collection_name")
    
    @property
    def hasjacobian(self) -> bool:
        return self._var.hasjacobian
    
    @property
    def normalized_blocks(self) -> List[str]:
        return self._var.normalized_blocks
    
    @property
    def normalized_by_err(self) -> bool:
        return True

    @property
    def jac_details(self) -> dict:
        return self._var.jac_details

class _ExtractCovarianceMatrix(VariableBase):
    def __init__(self, variable : PrebinnedVariableProtocol):
        self._var = variable

    @property
    def _natural_centerline(self):
        return 0

    @property
    def columns(self):
        return []
    
    @property
    def prebinned(self) -> bool:
        return True
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, _ExtractCovarianceMatrix):
            return False
        return (self._var == other._var)
    
    @property 
    def key(self):
        return "Cov(%s)" % self._var.key

    def set_collection_name(self, collection_name):
        raise ValueError("Prebinned Variables do not support set_collection_name")
    
    def evaluate(self, dataset, cut):
        if not isinstance(cut, PrebinnedOperationProtocol):
            raise ValueError("ExtractCovarianceMatrix requires a PrebinnedOperationProtocol cut")

        evaluated = self._var.evaluate(dataset, cut)
        if isinstance(dataset.binning, ArbitraryGenRecoBinning):
            return evaluated
        else:
            hist, cov, _, _ = maybe_valcov_to_definitely_valcov(evaluated)

            if cov is None:
                raise RuntimeError("ExtractCovarianceMatrix needs covariance!!")
            
            return cov
    
    @property
    def hasjacobian(self) -> bool:
        return self._var.hasjacobian
    
    @property
    def normalized_blocks(self) -> List[str]:
        return self._var.normalized_blocks
    
    @property
    def normalized_by_err(self) -> bool:
        return self._var.normalized_by_err
    
    @property
    def jac_details(self) -> dict:
        return self._var.jac_details

def strip_variable(variable : PrebinnedVariableProtocol) -> Tuple[PrebinnedVariableProtocol, dict]:
    if isinstance(variable, BasicPrebinnedVariable):
        return variable, {}
    elif isinstance(variable, WithJacobian):
        print("Stripping jacobian from variable")
        subvar, details = strip_variable(variable._var)
        details['jac_details'] = variable.jac_details
        return subvar, details
    elif isinstance(variable, NormalizePerBlock):
        print("Stripping NormalizePerBlock from variable")
        subvar, details = strip_variable(variable._var)
        details['normalized_blocks'] = variable.normalized_blocks
        return subvar, details
    elif isinstance(variable, DivideOutProfile):
        subvar, details = strip_variable(variable._var)
        details['profiled_blocks'] = variable.normalized_blocks
        return subvar, details
    else:
        raise ValueError("Unknown variable type %s"%type(variable))