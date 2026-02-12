
from typing import Sequence, reveal_type
from simonplot.cut.Cut import AndCuts, ConcatCut, NoCut
from simonplot.typing.Protocols import CutProtocol

def common_cuts_(cut1 : CutProtocol, cut2 : CutProtocol) -> CutProtocol:
    if cut1 == cut2:
        return cut1
    
    if isinstance(cut1, ConcatCut):
        cut1 = cut1.keycut
    if isinstance(cut2, ConcatCut):
        cut2 = cut2.keycut

    if isinstance(cut1, AndCuts) and not isinstance(cut2, AndCuts):
        common = []
        for c1 in cut1._cuts:
            if c1 == cut2:
                common.append(c1)
        if len(common) == 0:
            return NoCut()
        elif len(common) == 1:
            return common[0]
        else:
            return AndCuts(*common)
        
    elif isinstance(cut2, AndCuts) and not isinstance(cut1, AndCuts):
        return common_cuts_(cut2, cut1)
    
    elif isinstance(cut1, AndCuts) and isinstance(cut2, AndCuts):
        c1s = list(cut1._cuts)
        c2s = list(cut2._cuts)

        common = []
        for c1 in c1s:
            for c2 in c2s:
                if c1 == c2:
                    common.append(c1)
        
        if len(common) == 0:
            return NoCut()
        elif len(common) == 1:
            return common[0]
        else:
            return AndCuts(common)

    elif cut1 == cut2:
        return cut1
    else:  
        return NoCut()

def common_cuts(cuts : CutProtocol | Sequence[CutProtocol]) -> CutProtocol:
    #if cuts is not a sequence, just return it
    if not isinstance(cuts, Sequence):
        return cuts
    
    if len(cuts) == 0:
        return NoCut()
    elif len(cuts) == 1:
        return cuts[0]
    else:
        common = cuts[0]
        for cut in cuts[1:]:
            common = common_cuts_(common, cut)
        return common
