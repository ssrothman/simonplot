from .Abstract import AbstractPrebinnedDataset
from simon_mpl_util.util.AribtraryBinning import ArbitraryBinning
import numpy as np
from typing import List

class ValCovPairDataset(AbstractPrebinnedDataset):
    def __init__(self, key : str, data : tuple[np.ndarray, np.ndarray], binning : ArbitraryBinning):
        super().__init__(key, data, binning)

    @property
    def values(self):
        return self._data[0]
    
    @property
    def cov(self):
        return self._data[1]

    @property
    def is_stack(self) -> bool:
        return False
    
    def project(self, axes : List[str]):
        result = self.values
        projbinning = self._binning
        for ax in axes:
            result, projbinning = projbinning.project_out(result, ax)

        covresult = self.cov
        b2 = self._binning
        for ax in axes:
            covresult, b2 = b2.project_out_cov2d(covresult, ax)

        return result, covresult

    def slice(self, edges):
        result = self._binning.get_slice(self.values, **edges)
        covresult = self._binning.get_slice_cov2d(self.cov, **edges)
        return result, covresult

class CovmatDataset(AbstractPrebinnedDataset):
    def __init__(self, key:str, covmat : np.ndarray, binning : ArbitraryBinning):
        super().__init__(key, covmat, binning)

        self._covmat = covmat
    
    @property
    def covmat(self):
        return self._covmat
    
    @property
    def is_stack(self) -> bool:
        return False
    
    def project(self, axes : List[str]):
        result = self.covmat
        projbinning = self._binning
        for ax in axes:
            result, projbinning = projbinning.project_out_cov2d(result, ax)

        return result

    def slice(self, edges : dict):
        result = self._binning.get_slice_cov2d(self.covmat, **edges)
        return result