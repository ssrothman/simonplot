from simonplot.typing.Protocols import  PrebinnedDatasetAccessProtocol, UnbinnedDatasetAccessProtocol
from simonpy.AbitraryBinning import ArbitraryBinning

from abc import ABC, abstractmethod

class CutBase(ABC):
    @property
    @abstractmethod
    def prebinned(self) -> bool:
        raise NotImplementedError()

    @property
    @abstractmethod
    def columns(self):
        raise NotImplementedError()

    @abstractmethod
    def evaluate(self, dataset):
        raise NotImplementedError()

    @property
    @abstractmethod
    def key(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def _auto_label(self):
        raise NotImplementedError()

    @abstractmethod
    def __eq__(self, other):
        #error message says what subclass raised the error
        raise NotImplementedError("Equality operator not implemented for subclass %s"%(type(self).__name__))
    
    @abstractmethod
    def set_collection_name(self, collection_name):
        raise NotImplementedError()

    @property
    def label(self):
        if hasattr(self, "_label"):
            return self._label
        else:
            return self._auto_label

    def override_label(self, label):
        self._label = label

    def clear_override_label(self):
        del self._label

class UnbinnedCutBase(CutBase):
    def ensure_valid_dataset(self, dataset):
        if not isinstance(dataset, UnbinnedDatasetAccessProtocol):
            raise TypeError("UnbinnedCut can only be applied to UnbinnedDataset")
        return dataset

    @property
    def prebinned(self) -> bool:
        return False

class PrebinnedOperationBase(CutBase):
    @property
    def prebinned(self) -> bool:
        return False

    @property #override
    def columns(self):
        return []    
    
    def ensure_valid_dataset(self, dataset):
        if not isinstance(dataset, PrebinnedDatasetAccessProtocol):
            raise TypeError("PrebinnedCut can only be applied to PrebinnedDataset")
        return dataset

    def resulting_binning(self, binning : ArbitraryBinning) -> ArbitraryBinning:
        if not hasattr(self, '_resulting_binning'):
            self._resulting_binning = self._compute_resulting_binning(binning)

        return self._resulting_binning
        
    def _compute_resulting_binning(self, binning : ArbitraryBinning) -> ArbitraryBinning:
        raise NotImplementedError("PrebinnedOperation subclasses must implement compute_resulting_binning method")
    
    def set_collection_name(self, collection_name):
         raise ValueError("PrebinnedOperations do not support collection names")