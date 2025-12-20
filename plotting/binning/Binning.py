from simon_mpl_util.plotting.util.config import config, lookup_axis_label
from simon_mpl_util.plotting.variable.Variable import AbstractVariable
from simon_mpl_util.plotting.plottables.Datasets import AbstractDataset
from simon_mpl_util.plotting.cut.Cut import AbstractCut
from simon_mpl_util.plotting.cut.PrebinnedCut import PrebinnedOperation

from simon_mpl_util.util.AribtraryBinning import ArbitraryBinning

import hist
import awkward as ak
import numpy as np

from typing import Union, List

def transform_from_string(str : Union[str, None]) -> Union[hist.axis.transform.AxisTransform, None]:
    if str is None or str.lower() == "none":
        return None
    else:
        return getattr(hist.axis.transform, str)

class AbstractBinning:
    def build_axis(self, variable : AbstractVariable) -> hist.axis.AxesMixin:
        raise NotImplementedError()
    
    def build_default_axis(self, variable: AbstractVariable) -> hist.axis.AxesMixin:
        raise NotImplementedError()
    
    def build_auto_axis(self, 
                        variables: List[AbstractVariable], 
                        cuts: List[AbstractCut], 
                        datasets: List[AbstractDataset], 
                        transform: Union[str, None]=None) -> hist.axis.AxesMixin:
        raise NotImplementedError()

    def build_prebinned_axis(self, 
                             dataset : AbstractDataset,
                             cut : PrebinnedOperation) -> ArbitraryBinning:
        raise NotImplementedError()

class AutoIntCategoryBinning(AbstractBinning):
    def __init__(self, label_lookup : dict[str, str] = {}):
        self._label_lookup = label_lookup

    @property
    def label_lookup(self) -> dict[str, str]:
        return self._label_lookup

    def build_auto_axis(self, 
                        variables: List[AbstractVariable], 
                        cuts: List[AbstractCut], 
                        datasets: List[AbstractDataset], 
                        transform: Union[str, None]=None) -> hist.axis.AxesMixin:
        values = []
        lens = []
        for var, cut, dataset in zip(variables, cuts, datasets):
            needed_columns = list(set(var.columns + cut.columns))
            dataset.ensure_columns(needed_columns)
            v = var.evaluate(dataset)
            c = cut.evaluate(dataset)
            values.append(ak.flatten(v[c], axis=None)) # pyright: ignore[reportArgumentType]
            lens.append(len(values[-1]))

        all_values = ak.to_numpy(ak.flatten(values, axis=None)) # pyright: ignore[reportArgumentType]
        unique_values = np.unique(all_values)

        return hist.axis.IntCategory(
            unique_values.tolist(),
            name=variables[0].key,
            label=lookup_axis_label(variables[0].key),
            growth=False
        )

class AutoBinning(AbstractBinning):
    def __init__(self):
        self._force_low = None
        self._force_high = None

    def force_range(self, minval: Union[float, None], maxval: Union[float, None]):
        self._force_low = minval
        self._force_high = maxval

    def build_auto_axis(self, 
                        variables: List[AbstractVariable], 
                        cuts: List[AbstractCut], 
                        datasets: List[AbstractDataset], 
                        transform: Union[str, None]=None) -> hist.axis.AxesMixin:

        lens = []
        minvals = []
        maxvals = []
        dtypes = []
        for var, cut, dataset in zip(variables, cuts, datasets):
            needed_columns = list(set(var.columns + cut.columns))
            dataset.ensure_columns(needed_columns)
            v = var.evaluate(dataset)
            c = cut.evaluate(dataset)
            values = ak.to_numpy(ak.flatten(v[c], axis=None)) # pyright: ignore[reportArgumentType]
            values = values[np.isfinite(values)]
            lens.append(len(values))
            minvals.append(np.nanmin(values))
            maxvals.append(np.nanmax(values))
            dtypes.append(values.dtype)

        minval = np.min(minvals, axis=None)
        maxval = np.max(maxvals, axis=None)

        if minval == maxval:
            minval -= 0.5
            maxval += 0.5

        minlen = min(lens)

        dtype = dtypes[0]

        if self._force_low is not None:
            minval = self._force_low
        if self._force_high is not None:
            maxval = self._force_high

        if np.issubdtype(dtype, np.floating):
            #heuristic for a reasonable number of bins
            nbins = min(max(50, int(np.power(minlen, 1/3))), 150)

            return RegularBinning(
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
            return RegularBinning(
                nbins=nbins,
                low=minval-0.5,
                high=maxval+0.5,
                transform=transform
            ).build_axis(variables[0])

class DefaultBinning(AbstractBinning):
    def __init__(self):
        pass

    def build_default_axis(self, variable: AbstractVariable) -> hist.axis.AxesMixin:
        cfg = config['default_binnings'][variable.key]
        if cfg['type'] == 'regular':
            return RegularBinning(
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

class RegularBinning(AbstractBinning):
    def __init__(self, nbins : int, low : Union[float, int], high: Union[float, int], transform : Union[str, None]=None):
        self._nbins = nbins
        self._low = low
        self._high = high
        if type(transform) is str:
            self._transform = transform_from_string(transform)
        else:
            self._transform = None

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
    
    def build_axis(self, variable: AbstractVariable) -> hist.axis.AxesMixin:
        return hist.axis.Regular(
            self.nbins,
            self.low,
            self.high,
            transform=self.transform,
            name=variable.key,
            label=lookup_axis_label(variable.key)
        )
    
class ExplicitBinning(AbstractBinning):
    def __init__(self, edges: List[Union[float, int]]):
        self._edges = edges

    @property
    def edges(self) -> List[Union[float, int]]:
        return self._edges
    
    def build_axis(self, variable: AbstractVariable) -> hist.axis.AxesMixin:
        return hist.axis.Variable(
            self.edges,
            name=variable.key,
            label=lookup_axis_label(variable.key)
        )
    
