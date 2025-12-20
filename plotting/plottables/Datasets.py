from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.dataset as ds

import numpy as np
import awkward as ak

import hist
import matplotlib.axes

from simon_mpl_util.plotting.variable.Variable import AbstractVariable, PrebinnedVariable, ConstantVariable
from simon_mpl_util.plotting.cut.Cut import AbstractCut, NoCut
from simon_mpl_util.plotting.util.histplot import simon_histplot, simon_histplot_ratio, simon_histplot_arbitrary, simon_histplot_ratio_arbitrary

from simon_mpl_util.util.AribtraryBinning import ArbitraryBinning

from typing import List, Union, override

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

class UnbinnedDatasetStack(AbstractUnbinnedDataset):
    def __init__(self, key : str, datasets : list[AbstractUnbinnedDataset]):
        super().__init__(key)

        self._datasets = datasets

    def set_lumi(self, lumi):
        raise RuntimeError("DatasetStack.set_lumi: Cannot set lumi on DatasetStack!")
    
    def set_xsec(self, xsec):
        raise RuntimeError("DatasetStack.set_xsec: Cannot set xsec on DatasetStack!")
    
    def override_num_events(self, nevts):
        raise RuntimeError("DatasetStack.override_num_events: Cannot override number of events on DatasetStack!")

    @property
    def isMC(self):
        # DatasetStack is MC if all datasets are MC
        for d in self._datasets:
            if not d.isMC:
                return False
        return True
    
    @property
    def xsec(self):
        xsecs = [d.xsec for d in self._datasets]
        return np.sum(xsecs)
    
    @property
    def lumi(self):
        lumis = [d.lumi for d in self._datasets]
        return np.sum(lumis)
    
    def ensure_columns(self, columns):
        for d in self._datasets:
            d.ensure_columns(columns)

    def get_column(self, column_name, collection_name=None):
        results = []
        for d in self._datasets:
            results.append(d.get_column(column_name, collection_name=collection_name))
        
        return np.stack(results)

    def get_aknum_column(self, column_name):
        results = []
        for d in self._datasets:
            results.append(d.get_aknum_column(column_name))
        
        return np.stack(results)

    @property
    def num_rows(self):
        # return the minimum number of rows among the datasets
        # primary use case for this method is automatic binning
        # -> conservative when determining the number of bins to use
        return min([d.num_rows for d in self._datasets])

    @property
    def weight(self):
        return np.asarray([d.weight for d in self._datasets])

    def compute_weight(self, target_lumi):
        for d in self._datasets:
            d.compute_weight(target_lumi)

    def _fill_hist(self,
                  variable: AbstractVariable, 
                  cut: AbstractCut, 
                  weight : AbstractVariable,
                  axis : hist.axis.AxesMixin):
       
        for d in self._datasets:
            d._fill_hist(variable, cut, weight, axis)

        self.H = hist.Hist(
            axis, # pyright: ignore[reportArgumentType]
            storage=hist.storage.Weight()
        )

        for i in range(len(self._datasets)):
            self.H += self._datasets[i].H

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

        if fillbetween is not None and not own_style:
            raise ValueError("fillbetween is only supported when own_style is True")

        if own_style and fillbetween is None:
            mpl_kwargs['label'] = self.label
            mpl_kwargs['color'] = self.color

        prev = fillbetween
        if prev is not None:
            for d in self._datasets:
                (artist, vals), _ = d._plot_histogram(
                    variable, cut, weight, axis,
                    density, ax,
                    own_style=True,
                    fillbetween = prev,
                    **mpl_kwargs
                )
                prev = vals

            return (artist, vals), self.H # pyright: ignore[reportPossiblyUnboundVariable]
        else:        

            return simon_histplot(
                self.H, 
                ax = ax,
                density=density,
                fillbetween = fillbetween,
                **mpl_kwargs
            ), self.H
    
class NanoEventsDataset(AbstractUnbinnedDataset):
    def __init__(self, key, fname, **options):
        super().__init__(key)

        #suppress warnings
        NanoAODSchema.warn_missing_crossrefs = False

        import coffea
        version = coffea._version.version_tuple
        if int(version[0]) >= 2025 and int(version[1]) >= 11 and int(version[2]) >= 0:
            options['mode'] = 'virtual'
        else:
            options['delayed'] = False

        self._events = NanoEventsFactory.from_root(
            fname,
            **options 
        ).events()
    
    def ensure_columns(self, columns):
        # NanoEvents loads all columns on demand, so nothing to do here
        pass

    def get_column(self, column_name, collection_name=None):
        if '.' in column_name:
            raise ValueError("NanoEventsDataset.get_column: column_name '%s' contains '.'! Instead use collection_name argument."%(column_name))
        
        if collection_name is not None:
            return ak.materialize(self._events[collection_name][column_name])
        else:
            return ak.materialize(self._events[column_name])
    
    def get_aknum_column(self, column_name):
        return ak.to_numpy(ak.num(self._events[column_name]))
    
    @property
    def num_rows(self):
        return len(self._events)
    
class ParquetDataset(AbstractUnbinnedDataset):
    def __init__(self, key, path):
        super().__init__(key)

        self._dataset = ds.dataset(path, format="parquet")
        
    def ensure_columns(self, columns):
        has_everything = True
        if hasattr(self, '_table'):
            for col in columns:
                if col not in self._table.column_names:
                    has_everything = False
                    break
        else:
            has_everything = False

        if not has_everything:
            self._table = self._dataset.to_table(columns=columns)
    
    def get_column(self, column_name, collection_name=None):
        if collection_name is not None:
            raise NotImplementedError("ParquetDataset does not support collection_name argument")
        
        if not hasattr(self, '_table'):
            raise RuntimeError("ParquetDataset.ensure_columns must be called before get_column")
        
        if column_name not in self._table.column_names:
            raise RuntimeError("Column %s not loaded! Call ensure_columns() first"%column_name)

        return self._table[column_name].to_numpy()
    
    def get_aknum_column(self, column_name):
        raise NotImplementedError("ParquetDataset does not support ak.num columns")

    @property
    def num_rows(self):
        if hasattr(self, '_table'):
            return self._table.num_rows
        else:
            return self._dataset.count_rows()
    
    #extra properties for parquetdatasets for utility
    @property
    def files(self):
        return self._dataset.files
    
    @property 
    def filesystem(self):
        return self._dataset.filesystem
    
    @property
    def schema(self):
        return self._dataset.schema
    
class PrebinnedDataset(AbstractDataset):
    def __init__(self, key : str, values : np.ndarray, cov : np.ndarray, binning : ArbitraryBinning):
        super().__init__(key)
        self._values = values
        self._cov = cov
        self._binning = binning

    @property
    def values(self):
        return self._values
    
    @property
    def cov(self):
        return self._cov

    @property
    def binning(self):
        return self._binning

    def project(self, axes : List[str]):
        result = self._values
        projbinning = self._binning
        for ax in axes:
            result, projbinning = projbinning.project_out(result, ax)

        covresult = self._cov
        b2 = self._binning
        for ax in axes:
            covresult, b2 = b2.project_out_cov2d(covresult, ax)

        return result, covresult, projbinning

    def slice(self, edges):
        result = self._binning.get_slice(self._values, **edges)
        covresult = self._binning.get_slice_cov2d(self._cov, **edges)
        return result, covresult
    
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

        if not isinstance(weight, ConstantVariable):
            raise RuntimeError("PrebinnedDataset._plot_histogram: Cannot apply event weights to prebinned dataset! The weights have to be baked into the histogram when it is built!")
        
        if not isinstance(variable, PrebinnedVariable):
            raise TypeError("PrebinnedDataset._plot_histogram: variable must be a PrebinnedOperation")
    
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
