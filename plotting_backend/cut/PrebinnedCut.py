from .Cut import AbstractCut
from typing import List
from ..datasets import AbstractDataset, PrebinnedDataset
from ..util.AribtraryBinning import ArbitraryBinning
from ..util.SetupConfig import lookup_axis_label
from ..util import strip_units
import numpy as np

class PrebinnedOperation(AbstractCut):
    @property
    def columns(self):
        return []    
    
    def resulting_binning(self, binning : ArbitraryBinning) -> ArbitraryBinning:
        if not hasattr(self, '_resulting_binning'):
            self._resulting_binning = self._compute_resulting_binning(binning)

        return self._resulting_binning
        
    def _compute_resulting_binning(self, binning : ArbitraryBinning) -> ArbitraryBinning:
        raise NotImplementedError("PrebinnedOperation subclasses must implement compute_resulting_binning method")
    
class NoopOperation(PrebinnedOperation):
    @property
    def key(self):
        return "NOOP"
    
    def _auto_plottext(self):
        return ''

    def __eq__(self, other):
        return isinstance(other, NoopOperation)

    def evaluate(self, dataset):
        if not isinstance(dataset, PrebinnedDataset):
            raise TypeError("NoopOperation can only be applied to PrebinnedDataset")
        
        return dataset.values, dataset.cov

    def _compute_resulting_binning(self, binning : ArbitraryBinning) -> ArbitraryBinning:
        return binning

class ProjectionOperation(PrebinnedOperation):
    def __init__(self, axes : List[str]):
        self._axes = axes

    @property
    def key(self):
        return "PROJECT(%s)" % "-".join(str(ax) for ax in self._axes)

    def _auto_plottext(self):
        names = []
        for ax in self._axes:
            names.append(strip_units(lookup_axis_label(ax)))
        return 'Integrated over %s' % ", ".join(names)

    def __eq__(self, other):
        if not isinstance(other, ProjectionOperation):
            return False
        return self._axes == other._axes

    def evaluate(self, dataset):
        if not isinstance(dataset, PrebinnedDataset):
            raise TypeError("ProjectionOperation can only be applied to PrebinnedDataset")
        
        return dataset.project(self._axes)[:2]

    def _compute_resulting_binning(self, binning : ArbitraryBinning) -> ArbitraryBinning:
        result = binning
        empty_data = np.empty(binning.total_size)
        for ax in self._axes:
            empty_data, result = result.project_out(empty_data, ax)
        return result

class SliceOperation(PrebinnedOperation):
    def __init__(self, edges):
        self._edges = edges

    @property
    def key(self):
        slicestr = ''
        for name, edges in self._edges.items():
            slicestr+='%s-%0.3gto%0.3g_' % (name, edges[0], edges[1])
        if slicestr[-1] == '_':
            slicestr = slicestr[:-1]
        return "SLICE(%s)" % slicestr
    
    def _auto_plottext(self):
        texts = []
        for name, edges in self._edges.items():
            low = edges[0]
            high = edges[1]
            label = strip_units(lookup_axis_label(name))
            if low == -np.inf:
                texts.append('%s $< %0.3g$' % (label, high))
            elif high == np.inf:
                texts.append('%s $> %0.3g$' % (label, low))
            else:
                texts.append('$%0.3g < $%s$ < %0.3g$' % (low, label, high))
        return '\n'.join(texts)

    def __eq__(self, other):
        if not isinstance(other, SliceOperation):
            return False
        return self.key == other.key
    
    def evaluate(self, dataset):
        if not isinstance(dataset, PrebinnedDataset):
            raise TypeError("SliceOperation can only be applied to PrebinnedDataset")
        
        return dataset.slice(self._edges)
    
    def _compute_resulting_binning(self, binning : ArbitraryBinning) -> ArbitraryBinning:
        return binning.get_sliced_binning(**self._edges)

class ProjectAndSliceOperation(PrebinnedOperation):
    def __init__(self, axes : List[str], edges):
        self._projection = ProjectionOperation(axes)
        self._slice = SliceOperation(edges)

    @property
    def key(self):
        return "%s-%s" % (self._projection.key, self._slice.key)

    def __eq__(self, other):
        if not isinstance(other, ProjectAndSliceOperation):
            return False
        return self._projection == other._projection and self._slice == other._slice

    def _auto_plottext(self):
        texts = []
        proj_text = self._projection._auto_plottext()
        slice_text = self._slice._auto_plottext()
        if proj_text != '':
            texts.append(proj_text)
        if slice_text != '':
            texts.append(slice_text)
        return '\n'.join(texts)

    def evaluate(self, dataset):
        if not isinstance(dataset, PrebinnedDataset):
            raise TypeError("ProjectAndSliceOperation can only be applied to PrebinnedDataset")
        
        proj_dset = PrebinnedDataset(
            "TMP",
            *self._projection.evaluate(dataset),
            self._projection.resulting_binning(dataset.binning)
        )
        return self._slice.evaluate(proj_dset)
    
    def _compute_resulting_binning(self, binning : ArbitraryBinning) -> ArbitraryBinning:
        proj_binning = self._projection.resulting_binning(binning)
        slice_binning = self._slice._compute_resulting_binning(proj_binning)
        return slice_binning
