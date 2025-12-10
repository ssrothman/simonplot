from .SetupConfig import config, lookup_axis_label
import hist
import awkward as ak
import numpy as np

from .Variable import AbstractVariable
from .Cut import AbstractCut
from .datasets import AbstractDataset
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

class AutoBinning(AbstractBinning):
    def __init__(self):
        pass

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
            values.append(ak.flatten(v[c], axis=None))
            lens.append(len(values[-1]))

        all_values = ak.to_numpy(ak.flatten(values, axis=None))
        minval = np.nanmin(all_values, axis=None)
        maxval = np.nanmax(all_values, axis=None)

        minlen = min(lens)

        dtype = all_values.dtype
        if np.issubdtype(dtype, np.floating):
            #heuristic for a reasonable number of bins
            nbins = max(50, int(np.sqrt(minlen)))
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
            self._transform = transform

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