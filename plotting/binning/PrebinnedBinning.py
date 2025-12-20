from .Binning import AbstractBinning
from simon_mpl_util.plotting.plottables.Datasets import AbstractDataset, PrebinnedDataset
from simon_mpl_util.plotting.cut.PrebinnedCut import PrebinnedOperation

from simon_mpl_util.util.AribtraryBinning import ArbitraryBinning

class PrebinnedBinning(AbstractBinning):
    def build_prebinned_axis(self,
                             dataset : AbstractDataset,
                             cut : PrebinnedOperation) -> ArbitraryBinning:
        
        if not isinstance(dataset, PrebinnedDataset):
            raise TypeError("PrebinnedBinning can only build axes from PrebinnedDataset datasets")

        return cut.resulting_binning(dataset.binning)