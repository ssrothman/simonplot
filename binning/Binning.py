from simonplot.config import config, lookup_axis_label

from simonplot.typing.Protocols import BaseDatasetProtocol, PrebinnedDatasetProtocol, PrebinnedDatasetProtocol, UnbinnedDatasetProtocol, VariableProtocol, CutProtocol, PrebinnedOperationProtocol, BinningKind

from simonpy.AbitraryBinning import ArbitraryBinning

import hist
import awkward as ak
import numpy as np

from typing import Any, Union, List

from .BinningBase import BinningBase

def transform_from_string(str : Union[str, None]) -> Union[hist.axis.transform.AxisTransform, None]:
    if str is None or str.lower() == "none":
        return None
    else:
        return getattr(hist.axis.transform, str)

class AutoIntCategoryBinning(BinningBase):
    def __init__(self, label_lookup : dict[str, str] = {}):
        self._label_lookup = label_lookup

    @property
    def label_lookup(self) -> dict[str, str]:
        return self._label_lookup

    @property
    def has_custom_labels(self) -> bool:
        return True
    
    @property
    def kind(self) -> BinningKind:
        return BinningKind.AUTO
    
    def build_auto_axis(self, 
                        variables: List[VariableProtocol], 
                        cuts: List[CutProtocol], 
                        datasets: List[Any], 
                        transform: Union[str, None]=None) -> hist.axis.AxesMixin:
        values = []
        for var, cut, dataset in zip(variables, cuts, datasets):
            values.append(dataset.get_unique(var, cut))

        all_values = ak.to_numpy(ak.flatten(values, axis=None)) 
        unique_values = np.unique(all_values) 

        return hist.axis.IntCategory(
            unique_values.tolist(),
            name=variables[0].key,
            label=lookup_axis_label(variables[0].key),
            growth=False
        )

class AutoBinning(BinningBase):
    def __init__(self):
        self._force_low = None
        self._force_high = None

    @property
    def has_custom_labels(self) -> bool:
        return False
    
    @property
    def label_lookup(self) -> dict[str, str]:
        return {}

    @property
    def kind(self) -> BinningKind:
        return BinningKind.AUTO
    
    def force_range(self, minval: Union[float, None], maxval: Union[float, None]):
        self._force_low = minval
        self._force_high = maxval

    def build_auto_axis(self, 
                        variables: List[VariableProtocol], 
                        cuts: List[CutProtocol], 
                        datasets: List[BaseDatasetProtocol], 
                        transform: Union[str, None]=None) -> hist.axis.AxesMixin:

        lens = []
        minvals = []
        maxvals = []
        dtypes = []
        for var, cut, dataset in zip(variables, cuts, datasets):
            needed_columns = list(set(var.columns + cut.columns))

            #dataset.ensure_columns(needed_columns)
            minval, minval2, maxval, dtype = dataset.get_range(var, cut)
            if transform == 'log':
                minvals.append(minval2)
            else:
                minvals.append(minval)

            maxvals.append(maxval)
            lens.append(dataset.num_rows)
            dtypes.append(dtype)

        minval = np.min(minvals, axis=None)
        maxval = np.max(maxvals, axis=None)
       
        dtype = dtypes[0]

        if np.issubdtype(dtype, np.bool_):
            maxval = int(maxval)
            minval = int(minval)
        elif np.issubdtype(dtype, np.floating) and minval == maxval:
            minval = float(minval) - 0.5
            maxval = float(maxval) + 0.5

        minlen = min(lens)
        
        if self._force_low is not None:
            minval = self._force_low
        if self._force_high is not None:
            maxval = self._force_high

        if np.issubdtype(dtype, np.floating):
            #heuristic for a reasonable number of bins
            nbins = min(max(20, int(np.power(minlen, 1/3))//2), 100)

            return BasicBinning(
                nbins=nbins,
                low=minval,
                high=maxval,
                transform=transform
            ).build_axis(variables[0])
        else:
            #one bin for each integer value                
            nbins = int(maxval - minval + 1)

            if transform is not None:
                raise ValueError("Cannot use transform in AutoBinning with non-floating point variable")
            return BasicBinning(
                nbins=nbins,
                low=minval-0.5,
                high=maxval+0.5,
                transform=transform
            ).build_axis(variables[0])

class DefaultBinning(BinningBase):
    def __init__(self):
        pass

    @property
    def has_custom_labels(self) -> bool:
        return False
    
    @property
    def label_lookup(self) -> dict[str, str]:
        return {}

    @property
    def kind(self) -> BinningKind:
        return BinningKind.DEFAULT
    
    def build_default_axis(self, variable: VariableProtocol) -> hist.axis.AxesMixin:
        cfg = config['default_binnings'][variable.key]
        if cfg['type'] == 'regular':
            return BasicBinning(
                nbins=cfg['nbins'],
                low=cfg['low'],
                high=cfg['high'],
                transform=cfg.get('transform', None)
            ).build_axis(variable)
        elif cfg['type'] == 'explicit':
            return ExplicitBinning(
                edges=cfg['edges']
            ).build_axis(variable)
        else:
            raise ValueError("Unknown binning type: %s"%(cfg['type']))

class BasicBinning(BinningBase):
    def __init__(self, nbins : int, low : Union[float, int], high: Union[float, int], transform : Union[str, None]=None):
        self._nbins = nbins
        self._low = low
        self._high = high
        if type(transform) is str:
            self._transform = transform_from_string(transform)
        else:
            self._transform = None

    @property
    def has_custom_labels(self) -> bool:
        return False
    
    @property
    def label_lookup(self) -> dict[str, str]:
        return {}

    @property
    def kind(self) -> BinningKind:
        return BinningKind.BASIC
    
    @property
    def nbins(self) -> int:
        return self._nbins
    
    @property
    def low(self) -> Union[float, int]:
        return self._low
    
    @property
    def high(self) -> Union[float, int]:
        return self._high

    @property
    def transform(self) -> Union[hist.axis.transform.AxisTransform, None]:
        return self._transform
    
    def build_axis(self, variable: VariableProtocol) -> hist.axis.AxesMixin:
        return hist.axis.Regular(
            self.nbins,
            self.low,
            self.high,
            transform=self.transform,
            name=variable.key,
            label=lookup_axis_label(variable.key)
        )
    
class ExplicitBinning(BinningBase):
    def __init__(self, edges: List[Union[float, int]]):
        self._edges = edges

    @property
    def has_custom_labels(self) -> bool:
        return False
    
    @property
    def label_lookup(self) -> dict[str, str]:
        return {}

    @property
    def edges(self) -> List[Union[float, int]]:
        return self._edges
    
    @property
    def kind(self) -> BinningKind:
        return BinningKind.BASIC
    
    def build_axis(self, variable: VariableProtocol) -> hist.axis.AxesMixin:
        return hist.axis.Variable(
            self.edges,
            name=variable.key,
            label=lookup_axis_label(variable.key)
        )
    
class PrebinnedBinning(BinningBase):
    @property
    def kind(self) -> BinningKind:
        return BinningKind.PREBINNED
    
    @property
    def has_custom_labels(self) -> bool:
        return False
    
    @property
    def label_lookup(self) -> dict[str, str]:
        return {}

    def build_prebinned_axis(self,
                             dataset : Any,
                             cut : CutProtocol) -> ArbitraryBinning:
        
        if isinstance(cut, PrebinnedOperationProtocol):
            return cut.resulting_binning(dataset.binning)
        else:
            raise ValueError("Cut must be a PrebinnedOperationProtocol to build prebinned axis")