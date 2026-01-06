from ast import TypeVar
import numpy as np
import awkward as ak

from simonplot.cut.Cut import NoCut
from simonplot.util.histplot import simon_histplot, simon_histplot_ratio, simon_histplot_arbitrary, simon_histplot_ratio_arbitrary

from simonplot.typing.Protocols import BaseDatasetProtocol, HistplotMode, PrebinnedDatasetAccessProtocol, UnbinnedDatasetAccessProtocol, VariableProtocol, CutProtocol

from simonpy.AribtraryBinning import ArbitraryBinning

from typing import Any, List, Sequence, Tuple, Union, assert_never
import hist
import matplotlib.axes
import copy

from abc import ABC, abstractmethod

def call_histplot_function(H : Any, 
                           axis : Any,
                           ax : matplotlib.axes.Axes,
                           density : bool,
                           fillbetween : Any,
                           **mpl_kwargs) -> Tuple[Any, Any]:

    if isinstance(H, hist.Hist):
        return simon_histplot(
            H, 
            ax = ax,
            density=density,
            fillbetween = fillbetween,
            **mpl_kwargs
        )
    elif isinstance(H, tuple):
        if len(H) != 2:
            raise RuntimeError("call_histplot_function: Unsupported histogram type! [tuple with len != 2]")
        return simon_histplot_arbitrary(
            H[0], H[1],
            axis,
            ax = ax,
            density=density,
            fillbetween = fillbetween,
            **mpl_kwargs
        )
    else:
        raise RuntimeError("call_histplot_function: Unsupported histogram type! [neither hist.Hist nor tuple, but %s]"%type(H))

def call_histplot_ratio_function(H1 : Any, 
                                 H2 : Any,
                                 axis : Any,
                                 ax : matplotlib.axes.Axes,
                                 density : bool,
                                 **mpl_kwargs) -> Any:
    if isinstance(H1, hist.Hist):
        return simon_histplot_ratio(
            H1, H2,
            ax = ax,
            density=density,
            **mpl_kwargs
        )
    elif isinstance(H1, tuple):
        if len(H1) != 2 or len(H2) != 2:
            raise RuntimeError("call_histplot_ratio_function: Unsupported histogram type! [tuple with len != 2]")
        return simon_histplot_ratio_arbitrary(
            H1, H2,
            axis,
            ax = ax,
            density=density,
            **mpl_kwargs
        )
    else:
        raise RuntimeError("call_histplot_ratio_function: Unsupported histogram type! [neither hist.Hist nor tuple, but %s]"%type(H1))

def accumulate_H(H1 : Any, H2 : Any) -> Any:
    if type(H1) is not type(H2):
        raise RuntimeError("accumulate_H: Cannot accumulate histograms of different types! (%s vs %s)"%(type(H1), type(H2)))
    
    if isinstance(H1, hist.Hist):
        H1 += H2
        return H1
    elif isinstance(H1, tuple):
        if len(H1) != 2 or len(H2) != 2:
            raise RuntimeError("accumulate_H: Unsupported histogram type! [tuple with len != 2]")
        return (H1[0] + H2[0], H1[1] + H2[1])
    else:
        raise RuntimeError("accumulate_H: Unsupported histogram type! [neither hist.Hist nor tuple, but %s]"%type(H1))

class DatasetBase(ABC):
    _key : str

    @property
    @abstractmethod
    def is_stack(self) -> bool:
        raise NotImplementedError()

    def set_label(self, label):
        self._label = label

    def set_color(self, color):
        self._color = color

    @property
    def label(self):
        return self._label
    
    @property
    def key(self):
        return self._key

    @property
    def color(self):
        return self._color

    def override_num_events(self, nevts):
        self._override_nevts = nevts

    @property
    def num_events(self):
        if hasattr(self, '_override_nevts') and self._override_nevts is not None:
            return self._override_nevts
        else:
            if hasattr(self, 'num_rows'):
                return self.num_rows  # pyright: ignore[reportAttributeAccessIssue]

    def plot_hist_ratio(self,
                    H1 : Any,
                    H2 : Any,
                    axis : Any,
                    density : bool,
                    ax : matplotlib.axes.Axes,
                    own_style : bool,
                    **mpl_kwargs):
        
        if own_style:
            mpl_kwargs['color'] = self.color

        return call_histplot_ratio_function(
            H1, H2,
            axis,
            ax = ax,
            density = density,
            **mpl_kwargs
        )

class SingleDatasetBase(DatasetBase):
    _H : Any

    def estimate_yield(self, cut : CutProtocol, weight : VariableProtocol) -> float:
        needed_columns = list(set(cut.columns + weight.columns))
        
        self.ensure_columns(needed_columns)
        wgt = weight.evaluate(self, cut)  # pyright: ignore[reportArgumentType]
        total_yield = np.nansum(wgt) * self._weight

        return total_yield

    @abstractmethod
    def ensure_columns(self, columns: Sequence[str]):
        raise NotImplementedError()

    def get_range(self, var : VariableProtocol, cut : CutProtocol) -> Tuple[Any, Any, Any, np.dtype]:
        needed_columns = list(set(var.columns + cut.columns))
        
        self.ensure_columns(needed_columns)

        v = var.evaluate(self, cut) # pyright: ignore[reportArgumentType]
        values = ak.to_numpy(ak.flatten(v, axis=None)) # pyright: ignore[reportArgumentType]

        minval = np.nanmin(values)
        if len(values[values > 0]) == 0:
            minval2 = np.nan
        else:
            minval2 = np.nanmin(values[values > 0])
            
        maxval = np.nanmax(values)

        return (minval, minval2, maxval, values.dtype)

    def get_unique(self, var : VariableProtocol, cut : CutProtocol) -> np.ndarray:
        needed_columns = list(set(var.columns + cut.columns))
        
        self.ensure_columns(needed_columns)

        v = var.evaluate(self, cut) # pyright: ignore[reportArgumentType]
        values = ak.to_numpy(ak.flatten(v, axis=None)) # pyright: ignore[reportArgumentType]

        unique_values = np.unique(values) 

        return unique_values

    @property
    def is_stack(self) -> bool:
        return False

    def set_lumi(self, lumi):
        self._lumi = lumi
        self._isMC = False
        self._xsec = None

    def set_xsec(self, xsec):
        self._xsec = xsec
        self._isMC = True
        self._lumi = None

    @property
    def lumi(self) -> float:
        if self.isMC:
            raise RuntimeError("Dataset.lumi: Dataset is MC, no lumi defined!")
        if self._lumi is None:
            raise RuntimeError("Dataset.lumi: lumi not defined for dataset! Call set_lumi() first.")
        return self._lumi
    
    @property
    def xsec(self) -> float:
        if not self.isMC:
            raise RuntimeError("Dataset.xsec: Dataset is data, no xsec defined!")
        if self._xsec is None:
            raise RuntimeError("Dataset.xsec: xsec not defined for dataset! Call set_xsec() first.")
        return self._xsec

    @property
    def isMC(self):
        return self._isMC

    def compute_weight(self, target_lumi):
        if self.isMC:
            if self._xsec is None:
                raise RuntimeError("Dataset.compute_weight: xsec not defined for MC dataset! Call set_xsec() first.")
            
            self._weight = (target_lumi * 1000 * self._xsec) / self.num_events
        
        else:
            self._weight = 1.0

    def fill_hist(self,
                  variable: VariableProtocol, 
                  cut: CutProtocol, 
                  weight : VariableProtocol,
                  axis : Any) -> Any:
       
        if isinstance(self, UnbinnedDatasetAccessProtocol):
            needed_columns = list(set(variable.columns + cut.columns + weight.columns))
            self.ensure_columns(needed_columns)

            val = variable.evaluate(self, cut)
            wgt = weight.evaluate(self, cut)

            self._H = hist.Hist(
                axis,
                storage=hist.storage.Weight()
            )

            self._H.fill(
                ak.flatten(val, axis=None), 
                weight = self._weight * ak.flatten(wgt, axis=None) 
            )

        elif isinstance(self, PrebinnedDatasetAccessProtocol):
            cutresult = variable.evaluate(self, cut)

            if isinstance(cutresult, tuple) and len(cutresult) == 2:
                val, cov = cutresult
            else:
                raise RuntimeError("fill_hist: Cut must return prebinned (val, cov) pair for a prebinned dataset!")
            
            wgt = weight.evaluate(self, NoCut()) * self._weight

            val = val * wgt
            cov = cov * np.square(wgt)

            self._H = (val, cov)
        else:
            raise RuntimeError("fill_hist: Dataset does not implement UnbinnedDatasetAccessProtocol or PrebinnedDatasetAccessProtocol!")
        
        return self._H
    
    def plot_hist(self,
                variable: VariableProtocol, 
                cut: CutProtocol, 
                weight : VariableProtocol,
                axis : Any,
                density: bool,
                ax : matplotlib.axes.Axes,
                own_style : bool,
                mode : HistplotMode,
                _fillbetween : Union[float, None] = None,
                **mpl_kwargs) -> Tuple[Tuple[Any, Any], Any]:

        self.fill_hist(variable, cut, weight, axis)

        if own_style:
            mpl_kwargs['label'] = self.label
            mpl_kwargs['color'] = self.color

        if _fillbetween is not None:
            fbtw = _fillbetween
        elif mode != HistplotMode.ERRORBAR:
            fbtw = 0
        else:
            fbtw = None

        artist, vals = call_histplot_function(
            self._H, 
            axis,
            ax = ax,
            density=density,
            fillbetween = fbtw,
            **mpl_kwargs
        )
        return (artist, vals), self._H
    
class DatasetStackBase(DatasetBase):
    _datasets : Sequence[BaseDatasetProtocol]
    
    def estimate_yield(self, cut : CutProtocol, weight : VariableProtocol) -> float:
        return np.sum([d.estimate_yield(cut, weight) for d in self._datasets])

    def order_by_yield(self, cut : CutProtocol, weight : VariableProtocol) -> None:
        yields = [d.estimate_yield(cut, weight) for d in self._datasets]
        ordered_indices = np.argsort(yields)
        self._datasets = [self._datasets[i] for i in ordered_indices]

    def get_unique(self, var : VariableProtocol, cut : CutProtocol) -> np.ndarray:
        unique_values = np.unique(
            np.concatenate(
                [d.get_unique(var, cut) for d in self._datasets]
            )
        )
        return unique_values

    def get_range(self, var : VariableProtocol, cut : CutProtocol) -> Tuple[Any, Any, Any, np.dtype]:

        results = [d.get_range(var, cut) for d in self._datasets]
        minval = np.min([r[0] for r in results])
        minval2 = np.min([r[1] for r in results])
        maxval = np.max([r[2] for r in results])
        return (minval, minval2, maxval, results[0][0].dtype)

    @property
    def binning(self) -> ArbitraryBinning:
        if len(self._datasets) == 0:
            raise RuntimeError("DatasetStack.binning: No datasets in stack!")
        if not isinstance(self._datasets[0], PrebinnedDatasetAccessProtocol):
            raise RuntimeError("DatasetStack.binning: Datasets in stack are not prebinned datasets!")
        return self._datasets[0].binning

    @property
    def is_stack(self) -> bool:
        return True
    
    def set_lumi(self, lumi):
        raise RuntimeError("DatasetStack.set_lumi: Cannot set lumi on a dataset stack! Set lumi on individual datasets instead.")

    def set_xsec(self, xsec):
        raise RuntimeError("DatasetStack.set_xsec: Cannot set xsec on a dataset stack! Set xsec on individual datasets instead.")

    @property
    def lumi(self) -> float:
        if self.isMC:
            raise RuntimeError("Dataset.lumi: Dataset is MC, no lumi defined!")
        return np.sum([d.lumi for d in self._datasets])

    @property
    def xsec(self) -> float:
        if not self.isMC:
            raise RuntimeError("Dataset.xsec: Dataset is data, no xsec defined!")
        return np.sum([d.xsec for d in self._datasets])

    @property
    def num_rows(self):
        return np.min([d.num_rows for d in self._datasets])

    @property
    def isMC(self):
        return bool(np.all([d.isMC for d in self._datasets]))

    def compute_weight(self, target_lumi):
        for d in self._datasets:
            d.compute_weight(target_lumi)

    def fill_hist(self,
                  variable: VariableProtocol, 
                  cut: CutProtocol, 
                  weight : VariableProtocol,
                  axis : Any) -> Any:
       
        if 'NormalizePerBlock' in variable.key:
            # Kinda a weird edge case
            # When using a NormalizePerBlock variable, there is an implicit normalization
            # so stacks cannot just be summed up normally ....
            # for now just throw an error....
            # maybe later can implement some clever reweighting of the constituent blocks
            raise RuntimeError("DatasetStack.fill_hist: Cannot fill hist with NormalizePerBlock variable on a dataset stack!")

        for d in self._datasets:
            d.fill_hist(variable, cut, weight, axis)

        if len(self._datasets) == 0:
            raise RuntimeError("DatasetStack.fill_hist: No datasets in stack!")
        
        self.H = self._datasets[0].fill_hist(variable, cut, weight, axis)
        self.H = copy.deepcopy(self.H)

        for d in self._datasets[1:]:
            nextH = d.fill_hist(variable, cut, weight, axis)
            self.H = accumulate_H(self.H, nextH)

        return self.H

    def plot_hist(self,
                variable: VariableProtocol, 
                cut: CutProtocol, 
                weight : VariableProtocol,
                axis : Any,
                density: bool,
                ax : matplotlib.axes.Axes,
                own_style : bool,
                mode : HistplotMode,
                _fillbetween : Union[float, None] = None,
                **mpl_kwargs) -> Tuple[Any, Tuple[Any, Any]]:
        
        if len(self._datasets) == 0:
            raise RuntimeError("DatasetStack.plot_hist: No datasets in stack!")

        self.fill_hist(variable, cut, weight, axis)

        if _fillbetween is not None:
            fbtw = _fillbetween
        elif mode != HistplotMode.ERRORBAR:
            fbtw = 0
        else:
            fbtw = None

        if mode != HistplotMode.ERRORBAR and not own_style:
            raise ValueError("fillbetween is only supported when own_style is True")

        if own_style and mode != HistplotMode.STACK:
            mpl_kwargs['label'] = self.label
            mpl_kwargs['color'] = self.color

        if mode == HistplotMode.STACK:
            self.order_by_yield(cut, weight)

            prev = fbtw
            for d in self._datasets:
                (artist, vals), _ = d.plot_hist(
                    variable, cut, weight, axis,
                    density, ax,
                    own_style=True,
                    _fillbetween = prev,
                    mode = HistplotMode.FILL,
                    **mpl_kwargs
                )
                prev = vals

            return (artist, vals), self.H  # pyright: ignore[reportPossiblyUnboundVariable]
        else:   
            (artist, vals) = call_histplot_function(
                self.H, 
                axis,
                ax = ax,
                density=density,
                fillbetween = fbtw,
                **mpl_kwargs
            )   
            return (artist, vals), self.H