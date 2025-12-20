import numpy as np
import awkward as ak

from simon_mpl_util.plotting.util.histplot import simon_histplot, simon_histplot_ratio, simon_histplot_arbitrary, simon_histplot_ratio_arbitrary

from simon_mpl_util.plotting.variable.Abstract import AbstractVariable
from simon_mpl_util.plotting.cut.Abstract import AbstractCut

from simon_mpl_util.util.AribtraryBinning import ArbitraryBinning

from typing import Any, List, Union
import hist
import matplotlib.axes

class AbstractDataset:
    def __init__(self, key : str):
        self._lumi = None
        self._xsec = None
        self._isMC = True

        self._override_nevts = None

        self._weight = 1.0

        self._label = None
        self._color = None

        self._key = key

    @property
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

    def ensure_columns(self, columns):
        raise NotImplementedError()

    def get_column(self, column_name, collection_name=None):
        raise NotImplementedError()

    def get_aknum_column(self, column_name):
        raise NotImplementedError()
    
    @property
    def num_rows(self):
        raise NotImplementedError()

    def set_lumi(self, lumi):
        self._lumi = lumi
        self._isMC = False
        self._xsec = None

    def set_xsec(self, xsec):
        self._xsec = xsec
        self._isMC = True
        self._lumi = None

    def override_num_events(self, nevts):
        self._override_nevts = nevts

    @property
    def num_events(self):
        if self._override_nevts is not None:
            return self._override_nevts
        else:
            return self.num_rows

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

    @property
    def weight(self):
        return self._weight
    
    def compute_weight(self, target_lumi):
        if self.isMC:
            if self._xsec is None:
                raise RuntimeError("Dataset.compute_weight: xsec not defined for MC dataset! Call set_xsec() first.")
            
            self._weight = (target_lumi * 1000 * self._xsec) / self.num_events
        
        else:
            self._weight = 1.0

class AbstractUnbinnedDataset(AbstractDataset):
    def _fill_hist(self,
                  variable: AbstractVariable, 
                  cut: AbstractCut, 
                  weight : AbstractVariable,
                  axis : hist.axis.AxesMixin):
       
        needed_columns = list(set(variable.columns + cut.columns + weight.columns))
        self.ensure_columns(needed_columns)

        mask = cut.evaluate(self)
        val = variable.evaluate(self)
        wgt = weight.evaluate(self)

        #broadcast wgt to same shape as val
        wgt = ak.broadcast_arrays(val, wgt)[1]

        self.H = hist.Hist(
            axis, # pyright: ignore[reportArgumentType]
            storage=hist.storage.Weight()
        )
        self.H.fill(
            ak.flatten(val[mask], axis=None), # pyright: ignore[reportArgumentType]
            weight = self.weight * ak.flatten(wgt[mask], axis=None) # pyright: ignore[reportArgumentType]
        )

    def _plot_histogram(self,
                       variable: AbstractVariable, 
                       cut: AbstractCut, 
                       weight : AbstractVariable,
                       axis : hist.axis.AxesMixin,
                       density: bool,
                       ax : matplotlib.axes.Axes,
                       own_style : bool,
                       fillbetween : Union[float, None],
                       **mpl_kwargs):

        self._fill_hist(variable, cut, weight, axis)

        if own_style:
            mpl_kwargs['label'] = self.label
            mpl_kwargs['color'] = self.color

        return simon_histplot(
            self.H, 
            ax = ax,
            density=density,
            fillbetween = fillbetween,
            **mpl_kwargs
        ), self.H
    
    def _plot_ratio(self,
                    H1 : hist.Hist,
                    H2 : hist.Hist,
                    axis : hist.axis.AxesMixin,
                    density : bool,
                    ax : matplotlib.axes.Axes,
                    own_style : bool,
                    **mpl_kwargs):
        
        if own_style:
            mpl_kwargs['color'] = self.color

        return simon_histplot_ratio(
            H1, H2,
            ax = ax,
            density=density,
            **mpl_kwargs
        )

class AbstractPrebinnedDataset(AbstractDataset):
    def __init__(self, key : str, values : Any, binning : ArbitraryBinning):
        super().__init__(key)
        self._data = values
        self._binning = binning

    @property
    def data(self):
        return self._data

    @property
    def binning(self):
        return self._binning

    def project(self, axes : List[str]):
        raise NotImplementedError()

    def slice(self, edges):
        raise NotImplementedError()
    
    def _plot_histogram(self,
                       variable: AbstractVariable, 
                       cut: AbstractCut, 
                       weight : AbstractVariable,
                       axis : ArbitraryBinning,
                       density: bool,
                       ax : matplotlib.axes.Axes,
                       own_style : bool,
                       fillbetween : Union[float, None],
                       **mpl_kwargs):

        #if not isinstance(weight, ConstantVariable):
        #    raise RuntimeError("PrebinnedDataset._plot_histogram: Cannot apply event weights to prebinned dataset! The weights have to be baked into the histogram when it is built!")
        
        #if not isinstance(variable, PrebinnedVariable):
        #    raise TypeError("PrebinnedDataset._plot_histogram: variable must be a PrebinnedOperation")
    
        val, cov = cut.evaluate(self)
        wgt = weight.evaluate(self) * self.weight

        val = val * wgt
        cov = cov * np.square(wgt)

        self.H = val
        self.covH = cov

        if own_style:
            mpl_kwargs['label'] = self.label
            mpl_kwargs['color'] = self.color

        return simon_histplot_arbitrary(
            self.H, self.covH,
            axis,
            ax = ax,
            density=density,
            fillbetween = fillbetween,
            **mpl_kwargs
        ), (self.H, self.covH)
    
    def _plot_ratio(self,
                    H1 : np.ndarray,
                    H2 : np.ndarray,
                    axis : ArbitraryBinning,
                    density : bool,
                    ax : matplotlib.axes.Axes,
                    own_style : bool,
                    **mpl_kwargs):
        
        if own_style:
            mpl_kwargs['color'] = self.color

        return simon_histplot_ratio_arbitrary(
            H1, H2,
            axis,
            ax = ax,
            density = density,
            **mpl_kwargs
        )
