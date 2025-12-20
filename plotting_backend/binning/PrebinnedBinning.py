from .Binning import AbstractBinning
import hist
from ..util.AribtraryBinning import ArbitraryBinning
from ..variable.Variable import AbstractVariable, PrebinnedVariable
from ..datasets import AbstractDataset, PrebinnedDataset
from ..cut.PrebinnedCut import PrebinnedOperation

class PrebinnedBinning(AbstractBinning):
    def build_prebinned_axis(self,
                             dataset : AbstractDataset,
                             cut : PrebinnedOperation) -> ArbitraryBinning:
        
        if not isinstance(dataset, PrebinnedDataset):
            raise TypeError("PrebinnedBinning can only build axes from PrebinnedDataset datasets")

        return cut.resulting_binning(dataset.binning)