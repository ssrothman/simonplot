import matplotlib.axes
from ..plotting_backend.util.SetupConfig import config
from ..variable.Variable import AbstractVariable, variable_from_string, RatioVariable, DifferenceVariable, RelativeResolutionVariable
from ..cut.Cut import AbstractCut, common_cuts, NoCut
from ..plotting_backend.datasets import AbstractDataset
from ..binning.Binning import AbstractBinning, AutoBinning, DefaultBinning

from ..util import simon_histplot

from ..util import setup_canvas, add_cms_legend, savefig, ensure_same_length, add_text, draw_legend, make_oneax
from ..util import place_text
from ..plotting_backend.PlotStuff import AbstractPlotSpec

import hist
import matplotlib.pyplot as plt
import awkward as ak

from typing import List, Sequence, Union

def scatter_2d(varX_ : Union[AbstractVariable, List[AbstractVariable]], 
               varY_ : Union[AbstractVariable, List[AbstractVariable]], 
               cut_: Union[AbstractCut, List[AbstractCut]], 
               dataset_: Union[AbstractDataset, List[AbstractDataset]],
               labels_: Union[List[str], None] = None,
               extratext : Union[str, None] = None,
               isdata: bool = False,
               logx: bool = False,
               logy: bool = False,
               ensure_square_aspect: bool = False,
               notext : bool = False,
               ps : float = 1.0,
               output_path: Union[str, None] = None,
               legend_loc : Union[str, int, tuple] ='best',
               skip_empty : bool = True,
               add_stuff : Union[None, List[AbstractPlotSpec]] = None):
    
    if labels_ is None or len(labels_) == 1:
        nolegend = True
    else:
        nolegend = False

    if labels_ is None:
        labels_ = ['']

    varX, varY, cut, dataset, labels = ensure_same_length(varX_, varY_, cut_, dataset_, labels_)

    fig = setup_canvas()
    ax = make_oneax(fig)
    add_cms_legend(ax, isdata)

    for vX, vY, c, d, l in zip(varX, varY, cut, dataset, labels):
        _, _ = scatter_2d_(vX, vY, c, d, l, ax, ps=ps, skip_empty=skip_empty)

    if add_stuff is not None:
        for stuff in add_stuff:
            stuff.plot(ax)

    ax.set_xlabel(varX[0].label)
    ax.set_ylabel(varY[0].label)

    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')

    if ensure_square_aspect:
        ax.set_aspect('equal', adjustable='box')

    if not notext:
        add_text(ax, cut, extratext)

    draw_legend(ax, nolegend, loc=legend_loc)

    plt.tight_layout()

    if output_path is not None:
        savefig(fig, output_path)
    else:
        plt.show()
        
    plt.close(fig)

def scatter_2d_(varX: AbstractVariable, varY: AbstractVariable, 
                cut: AbstractCut, 
                dataset: AbstractDataset,
                label: str,
                ax : matplotlib.axes.Axes,
                skip_empty : bool = True,
                ps : float = 1.0):
    
    needed_columns = list(set(varX.columns + varY.columns + cut.columns))

    dataset.ensure_columns(needed_columns)

    x = varX.evaluate(dataset)
    y = varY.evaluate(dataset)
    c = cut.evaluate(dataset)

    xvals = ak.flatten(x[c], axis=None) # pyright: ignore[reportArgumentType]
    yvals = ak.flatten(y[c], axis=None) # pyright: ignore[reportArgumentType]

    if len(xvals) == 0 and skip_empty:
        return None, (xvals, yvals)

    return ax.scatter(
        xvals,
        yvals,
        s=ps,
        label=label,
        alpha=0.7,
        rasterized=True
    ), (xvals, yvals)