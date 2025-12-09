from .SetupConfig import config
from .Variable import AbstractVariable, variable_from_string, RatioVariable, DifferenceVariable, RelativeResolutionVariable
from .Cut import AbstractCut, common_cuts, NoCut
from .datasets import AbstractDataset
from .Binning import AbstractBinning

from .histplot import simon_histplot

from .util import setup_canvas, add_cms_legend, savefig
from .place_text import place_text

import hist
import matplotlib.pyplot as plt
import awkward as ak

from typing import List, Union

def plot_histogram(variable: Union[AbstractVariable, List[AbstractVariable]], 
                   cut: Union[AbstractCut, List[AbstractCut]], 
                   dataset: Union[AbstractDataset, List[AbstractDataset]],
                   binning : AbstractBinning,
                   labels: Union[List[str], None] = None,
                   extratext : Union[str, None] = None,
                   isdata: bool = False,
                   density: bool = False,
                   logx: bool = False,
                   logy: bool = False):
    
    if isinstance(cut, AbstractCut):
        cut = [cut]
    if isinstance(variable, AbstractVariable):
        variable = [variable]
    if isinstance(dataset, AbstractDataset):
        dataset = [dataset]

    maxlen = max(len(cut), len(variable), len(dataset))
    if len(cut) not in [1, maxlen]:
        raise ValueError("Length of cut list must be 1 or %d"%maxlen)
    if len(variable) not in [1, maxlen]:
        raise ValueError("Length of variable list must be 1 or %d"%maxlen)
    if len(dataset) not in [1, maxlen]:
        raise ValueError("Length of dataset list must be 1 or %d"%maxlen)
    
    if labels is not None and len(labels) != maxlen:
        raise ValueError("Length of labels list must be %d"%maxlen)

    if labels is None or len(labels) == 1:
        nolegend = True
    else:
        nolegend = False

    if labels is None:
        labels = [''] * maxlen

    if len(cut) == 1:
        cut = cut * maxlen
    if len(variable) == 1:
        variable = variable * maxlen
    if len(dataset) == 1:
        dataset = dataset * maxlen

    fig, ax = setup_canvas()
    add_cms_legend(ax, isdata)

    vals = []
    for v, c, d, l in zip(variable, cut, dataset, labels):
        _, value = plot_variable(v, c, d, binning, density, l, ax)
        vals.append(value)

    ax.set_xlabel(variable[0].label)
    if density:
        ax.set_ylabel('Density')
    else:
        ax.set_ylabel('Counts')

    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')

    if type(variable[0]) is RatioVariable:
        ax.axvline(1.0, color='k', linestyle='dashed')
    elif type(variable[0]) is DifferenceVariable or type(variable[0]) is RelativeResolutionVariable:
        ax.axvline(0.0, color='k', linestyle='dashed')

    ccut = common_cuts(cut)

    if type(ccut) is not NoCut:
        thetext = ccut.plottext
    else:
        thetext = ''

    if extratext is not None:
        thetext = extratext + '\n' + thetext

    thetext = thetext.strip()

    if thetext != '':
        place_text(ax, thetext, loc='best', fontsize=24, bbox_opts={
            'boxstyle': 'round,pad=0.3',
            'facecolor': 'white',
            'edgecolor': 'black',
            'alpha': 0.8
        })

    if not nolegend:
        ax.legend(
            fontsize=18, 
            loc='best',
            framealpha=0.8,
            borderpad=0.3,
            frameon=True
        )

    plt.tight_layout()
    savefig(fig, 'test')
    plt.close(fig)

def plot_variable(variable: AbstractVariable, 
                  cut: AbstractCut, 
                  dataset: AbstractDataset,
                  binning : AbstractBinning,
                  density: bool,
                  label: str,
                  ax : plt.Axes):
   
    needed_columns = list(set(variable.columns + cut.columns))
    dataset.ensure_columns(needed_columns)

    cut = cut.evaluate(dataset)
    val = variable.evaluate(dataset)

    axis = binning.build_axis(variable)
    H = hist.Hist(
        axis,
        storage=hist.storage.Weight()
    )
    H.fill(
        ak.flatten(val[cut], axis=None)
    )

    return simon_histplot(
        H, 
        ax = ax,
        density=density,
        label=label
    )