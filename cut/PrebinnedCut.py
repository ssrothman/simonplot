from simonplot.typing.Protocols import PrebinnedDatasetAccessProtocol
from simonplot.config import lookup_axis_label

from simonpy.text import strip_units
from simonpy.AbitraryBinning import ArbitraryBinning

import numpy as np

from typing import List, Sequence

from .CutBase import PrebinnedOperationBase

class NoopOperation(PrebinnedOperationBase):
    def __init__(self):
        pass #stateless

    @property
    def key(self):
        return "NOOP"
    
    @property
    def _auto_label(self):
        return ''

    def __eq__(self, other):
        return isinstance(other, NoopOperation)

    def evaluate(self, dataset):
        dataset = self.ensure_valid_dataset(dataset)   
        
        return dataset.data

    def _compute_resulting_binning(self, binning : ArbitraryBinning) -> ArbitraryBinning:
        return binning

class ProjectionOperation(PrebinnedOperationBase):
    def __init__(self, axes : Sequence[str]):
        self._axes = axes

    @property
    def key(self):
        return "PROJECT(%s)" % "-".join(str(ax) for ax in self._axes)

    @property
    def _auto_label(self):
        names = []
        for ax in self._axes:
            names.append(strip_units(lookup_axis_label(ax)))
        return 'Integrated over %s' % ", ".join(names)

    def __eq__(self, other):
        if not isinstance(other, ProjectionOperation):
            return False
        return self._axes == other._axes

    def evaluate(self, dataset):
        dataset = self.ensure_valid_dataset(dataset)   
        return dataset.project(self._axes)

    def _compute_resulting_binning(self, binning : ArbitraryBinning) -> ArbitraryBinning:
        result = binning
        empty_data = np.zeros(binning.total_size)
        for ax in self._axes:
            empty_data, result = result.project_out(empty_data, ax)
        return result

class SliceOperation(PrebinnedOperationBase):
    def __init__(self, edges : dict[str, Sequence]):
        self._edges = edges

    @property
    def key(self):
        slicestr = ''
        for name, edges in self._edges.items():
            slicestr+='%s-%0.3gto%0.3g_' % (name, edges[0], edges[1])
        if slicestr[-1] == '_':
            slicestr = slicestr[:-1]
        return "SLICE(%s)" % slicestr
    
    @property
    def _auto_label(self):
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
        dataset = self.ensure_valid_dataset(dataset)   
        return dataset.slice(self._edges)
    
    def _compute_resulting_binning(self, binning : ArbitraryBinning) -> ArbitraryBinning:
        return binning.get_sliced_binning(self._edges)

class ProjectAndSliceOperation(PrebinnedOperationBase):
    def __init__(self, axes : Sequence[str], edges: dict[str, Sequence]):
        self._projection = ProjectionOperation(axes)
        self._slice = SliceOperation(edges)

    @property
    def key(self):
        return "%s-%s" % (self._projection.key, self._slice.key)

    def __eq__(self, other):
        if not isinstance(other, ProjectAndSliceOperation):
            return False
        return self._projection == other._projection and self._slice == other._slice

    @property
    def _auto_label(self):
        texts = []
        proj_text = self._projection._auto_label
        slice_text = self._slice._auto_label
        if proj_text != '':
            texts.append(proj_text)
        if slice_text != '':
            texts.append(slice_text)
        return '\n'.join(texts)

    def evaluate(self, dataset):
        dataset = self.ensure_valid_dataset(dataset)   
        
        projdata = self._projection.evaluate(dataset)
        projbinning = self._projection.resulting_binning(dataset.binning)
        proj_dset = dataset._dummy_dset(projdata, projbinning)

        return self._slice.evaluate(proj_dset)
    
    def _compute_resulting_binning(self, binning : ArbitraryBinning) -> ArbitraryBinning:
        proj_binning = self._projection.resulting_binning(binning)
        slice_binning = self._slice._compute_resulting_binning(proj_binning)
        return slice_binning
