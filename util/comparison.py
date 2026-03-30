from typing import Literal
from xxlimited import Str

from networkx import density
import numpy as np
import hist
import awkward as ak

class ComparisonHistStruct:
    _SUPPORTED_MODES = Literal['ratio', 'difference']

    def __init__(self, hist1 : hist.Hist, hist2 : hist.Hist, mode : _SUPPORTED_MODES):
        self._hist1 = hist1
        self._hist2 = hist2
        self._mode = mode

        self._density = False

        if hist1.axes != hist2.axes:
            raise RuntimeError("ComparisonHistStruct: hist1 and hist2 must have the same axes!")

    def set_density(self, density : bool):
        self._density = density

    @property
    def H1(self):
        return self._hist1
    
    @property
    def H2(self):
        return self._hist2
    
    @property
    def mode(self):
        return self._mode
    
    @property
    def axes(self):
        return self._hist1.axes
    
    def __add__(self, other):
        raise NotImplementedError("ComparisonHistStruct does not support addition!")
    
    def __iadd__(self, other):
        raise NotImplementedError("ComparisonHistStruct does not support addition!")
    
    '''
    Mimick the hist.Hist() interface
    '''

    #statistic
    def values(self, flow=False):
        H1sum = self._hist1.values(flow=True).sum()
        H2sum = self._hist2.values(flow=True).sum()

        if self._mode == 'ratio':
            if self._density:
                return (H2sum / H1sum) * self._hist1.values(flow=flow) / self._hist2.values(flow=flow)
            else:
                return self._hist1.values(flow=flow) / self._hist2.values(flow=flow)

        elif self._mode == 'difference':
            if self._density:
                return self._hist1.values(flow=flow)/H1sum - self._hist2.values(flow=flow)/H2sum
            else:
                return self._hist1.values(flow=flow) - self._hist2.values(flow=flow)
        else:
            raise ValueError(f"Unsupported mode {self._mode} for ComparisonHistStruct! Supported modes are: 'ratio' and 'difference'.")

    
    #just return 0 for uncertainties atm
    def variances(self, flow=False):
        H1sum = self._hist1.values(flow=True).sum()
        H2sum = self._hist2.values(flow=True).sum()

        if self._mode == 'ratio':
            # error propagation for ratio A/B, assuming A and B are uncorrelated
            varA = self._hist1.variances(flow=flow)
            varB = self._hist2.variances(flow=flow)
            A = self._hist1.values(flow=flow)
            B = self._hist2.values(flow=flow)

            if varA is None:
                raise ValueError("ComparisonHistStruct: hist1 does not have variances, cannot compute variances for ratio mode!")
            if varB is None:
                raise ValueError("ComparisonHistStruct: hist2 does not have variances, cannot compute variances for ratio mode!")

            if not self._density:
                return np.square(1/B) * varA + np.square(A/(B*B)) * varB
            else:
                A_density = A / H1sum
                B_density = B / H2sum
                varA_density = varA / (H1sum*H1sum)
                varB_density = varB / (H2sum*H2sum)

                return np.square(1/B_density) * varA_density + np.square(A_density/(B_density*B_density)) * varB_density # this isn't quite right, but its close...

        elif self._mode == 'difference':
            # error propagation for difference A-B, assuming A and B are uncorrelated
            varA = self._hist1.variances(flow=flow)
            varB = self._hist2.variances(flow=flow)

            if varA is None:
                raise ValueError("ComparisonHistStruct: hist1 does not have variances, cannot compute variances for difference mode!")
            if varB is None:
                raise ValueError("ComparisonHistStruct: hist2 does not have variances, cannot compute variances for difference mode!")

            if not self._density:
                return varA + varB
            else:
                varA_density = varA / (H1sum*H1sum)
                varB_density = varB / (H2sum*H2sum)
                return varA_density + varB_density
        else:
            raise ValueError(f"Unsupported mode {self._mode} for ComparisonHistStruct! Supported modes are: 'ratio' and 'difference'.")