
from .CutBase import UnbinnedCutBase
from .NoCut import NoCut
from .LogicalCuts import AndCuts
from simonplot.typing.Protocols import CutProtocol, VariableProtocol, UnbinnedDatasetAccessProtocol, UnbinnedDatasetProtocol
import copy
from typing import Any, List, Sequence, Union
import numpy as np

class ConcatCut(UnbinnedCutBase):
    def __init__(self, *cuts, keycut=None):
        self._cuts = cuts

        if keycut is None:
            self._keycut = NoCut()
            print("Warning: ConcatCut has no keycut, so automatic labels will be blank")
        else:
            self._keycut = keycut

    @staticmethod
    def build_for_collections(cut : CutProtocol, collections_l : Sequence[str], unique_cuts_l : Union[None, Sequence[CutProtocol]]=None):
        if unique_cuts_l is not None and len(unique_cuts_l) != len(collections_l):
            raise ValueError("ConcatCut.build_for_collections: unique_cuts_l length does not match collections_l length")
        
        if unique_cuts_l is None:
            unique_cuts_l = [NoCut()] * len(collections_l)

        cuts : List[CutProtocol] = []
        for coll, ucut in zip(collections_l, unique_cuts_l):
            c = copy.deepcopy(cut)
            c.set_collection_name(coll)
            cuts.append(AndCuts([c, ucut]))
            
        return ConcatCut(*cuts, keycut=cut)

    @property
    def columns(self):
        cols = []
        for cut in self._cuts:
            cols += cut.columns
        return list(set(cols))

    @property
    def keycut(self):
        return self._keycut

    def evaluate(self, dataset):
        dataset = self.ensure_valid_dataset(dataset)   
        masks = [cut.evaluate(dataset) for cut in self._cuts]
        return np.concatenate(masks)

    @property
    def key(self):
        return self._keycut.key

    @property
    def _auto_label(self):
        return self._keycut.label

    def set_collection_name(self, collection_name):
        print("WARNING: overwriting collection name for all cuts in ConcatCut object")
        for cut in self._cuts:
            cut.set_collection_name(collection_name)
        self._keycut.set_collection_name(collection_name)

    def __eq__(self, other):
        if not isinstance(other, ConcatCut):
            return False

        return self._keycut == other._keycut
