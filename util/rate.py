import numpy as np
import hist
import awkward as ak

class RateStruct:
    def __init__(self, binary : np.ndarray | ak.Array, wrt : np.ndarray | ak.Array):
        self._binary = binary
        self._wrt = wrt
    
    @property
    def binary(self):
        return self._binary
    
    @property
    def wrt(self): 
        return self._wrt

# container for two histograms representing the pass and fail categories of a RateVariable
# needed for plotting rates with histplot
class RateHistStruct:
    def __init__(self, Hpass, Hfail):
        self._Hpass = Hpass
        self._Hfail = Hfail

        if Hpass.axes != Hfail.axes:
            raise RuntimeError("RateHistStruct: Hpass and Hfail must have the same axes!")

    @property
    def Hpass(self):
        return self._Hpass
    
    @property
    def Hfail(self):
        return self._Hfail

    @property
    def axes(self):
        return self._Hpass.axes

    def __add__(self, other):
        if not isinstance(other, RateHistStruct):
            raise RuntimeError("RateHistStruct.__add__: Can only add another RateHistStruct, but got %s!"%type(other))
        
        newHpass = self._Hpass + other._Hpass
        newHfail = self._Hfail + other._Hfail

        return RateHistStruct(newHpass, newHfail)
    
    def __iadd__(self, other):
        if not isinstance(other, RateHistStruct):
            raise RuntimeError("RateHistStruct.__iadd__: Can only add another RateHistStruct, but got %s!"%type(other))
        
        self._Hpass += other._Hpass
        self._Hfail += other._Hfail

        return self

    '''
    mimick the hist.Hist() interface
    '''

    # return rates
    def values(self, flow=False): 
        Ntotal = self._Hpass.values(flow=flow) + self._Hfail.values(flow=flow)
        Npass = self._Hpass.values(flow=flow)
        return Npass / Ntotal
    
    # error propagation for ratio Npass / (Npass + Nfail)
    # use formula from Appendix B.1.3 of the AN
    def variances(self, flow=False):
        Nfail = self._Hfail.values(flow=flow)
        Npass = self._Hpass.values(flow=flow)
        Ntotal = Npass + Nfail

        varPass = self._Hpass.variances(flow=flow)
        varFail = self._Hfail.variances(flow=flow)

        rate = Npass / Ntotal

        return np.square((1-rate)/Ntotal) * varPass + np.square(rate/Ntotal) * varFail

