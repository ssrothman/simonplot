from simonplot.typing.Protocols import PrebinnedDatasetAccessProtocol
from .DatasetBase import SingleDatasetBase

from simonpy.AbitraryBinning import ArbitraryBinning
import numpy as np
from typing import Sequence, Tuple

import uproot

class PrebinnedDatasetBase(SingleDatasetBase):
    _data : np.ndarray | Tuple[np.ndarray, np.ndarray]
    _binning : ArbitraryBinning

    @property
    def data(self):
        return self._data

    @property
    def binning(self):
        return self._binning

class ValCovPairDataset(PrebinnedDatasetBase):
    def __init__(self, 
                 key : str, 
                 color : str | None,
                 label : str | None,
                 data : tuple[np.ndarray, np.ndarray], 
                 binning : ArbitraryBinning,
                 isMC : bool = True):
        self._key = key
        self._color = color
        self._label = label

        self._data = data
        self._binning = binning
        
        self._isMC = isMC

    def ensure_columns(self, columns: Sequence[str]):
        pass

    @property
    def quantitytype(self):
        return "valcov"
    
    @property
    def values(self):
        return self._data[0]
    
    @property
    def cov(self):
        return self._data[1]
    
    def project(self, axes : Sequence[str]):
        result = self.values
        projbinning = self._binning
        for ax in axes:
            #print("Projecting out axis:", ax)  
            #print("result.sum() = ", np.sum(result))
            result, projbinning = projbinning.project_out(result, ax)

        covresult = self.cov
        b2 = self._binning
        for ax in axes:
            #print("Projecting out axis from cov:", ax)
            #print("covresult.sum() = ", np.sum(covresult))
            covresult, b2 = b2.project_out_cov2d(covresult, ax)

        return result, covresult

    def slice(self, edges):
        result = self._binning.get_slice(self.values, edges)
        covresult = self._binning.get_slice_cov2d(self.cov, edges)
        return result, covresult

    def _dummy_dset(self, data, binning) -> PrebinnedDatasetAccessProtocol:
        return ValCovPairDataset("", '', '', data, binning)
    
    @property
    def num_rows(self) -> int:
        return np.sum(self.data[0])
    

class PrebinnedRootHistogramDataset(PrebinnedDatasetBase):
    def __init__(self, 
                 key : str, 
                 path : str,
                 color : str | None,
                 label : str | None,
                 isMC : bool = True):
        self._key = key
        self._color = color
        self._label = label
        
        self._isMC = isMC

        self._H = uproot.open(path).to_hist() # type: ignore

    def ensure_columns(self, columns: Sequence[str]):
        pass

    @property
    def quantitytype(self):
        return "valcov"

    @property
    def values(self):
        return self._H.values(flow=True)
    
    @property
    def cov(self):
        return np.diagonal(self._H.variances(flow=True))
    
    def project(self, axes : Sequence[str]):
        raise NotImplementedError()
    
    def slice(self, edges):
        raise NotImplementedError()
    
    def _dummy_dset(self, data, binning) -> PrebinnedDatasetAccessProtocol:
        raise NotImplementedError()
    
    @property
    def num_rows(self) -> int:
        return np.sum(self.values)

class CovmatDataset(PrebinnedDatasetBase):
    def __init__(self, 
                 path : str,
                 key : str, 
                 color : str | None,
                 label : str | None,
                 binning : ArbitraryBinning,
                 isMC : bool = True):
        self._key = key
        self._color = color
        self._label = label

        self._binning = binning
        
        self._isMC = isMC
    
    @property
    def quantitytype(self):
        return "covariance"
        
    @property
    def covmat(self):
        if not isinstance(self._data, np.ndarray):
            raise ValueError("Data is not a covariance matrix!")
        return self._data
        
    def project(self, axes : Sequence[str]):
        result = self.covmat
        projbinning = self._binning
        for ax in axes:
            result, projbinning = projbinning.project_out_cov2d(result, ax)

        return result

    def slice(self, edges : dict):
        result = self._binning.get_slice_cov2d(self.covmat, edges)
        return result
    
    def ensure_columns(self, columns: Sequence[str]):
        pass

    def _dummy_dset(self, data, binning) -> PrebinnedDatasetAccessProtocol:
        return CovmatDataset("", '', '', data, binning)
    
    @property
    def num_rows(self) -> int:
        return np.sum(self.data)