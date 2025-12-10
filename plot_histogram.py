from .SetupConfig import config
from .Variable import AbstractVariable, variable_from_string, RatioVariable, DifferenceVariable, RelativeResolutionVariable
from .Cut import AbstractCut, common_cuts, NoCut
from .datasets import AbstractDataset
from .Binning import AbstractBinning, AutoBinning, DefaultBinning, AutoIntCategoryBinning

from .histplot import simon_histplot

from .util import setup_canvas, add_cms_legend, savefig, ensure_same_length, add_text, draw_legend

import hist
import matplotlib.pyplot as plt
import awkward as ak

from typing import List, Union

def plot_histogram(variable_: Union[AbstractVariable, List[AbstractVariable]], 
                   cut_: Union[AbstractCut, List[AbstractCut]], 
                   dataset_: Union[AbstractDataset, List[AbstractDataset]],
                   binning : AbstractBinning,
                   labels_: Union[List[str], None] = None,
                   extratext : Union[str, None] = None,
                   isdata: bool = False,
                   density: bool = False,
                   logx: bool = False,
                   logy: bool = False,
                   output_path: Union[str, None] = None):
    
    if labels_ is None or len(labels_) == 1:
        nolegend = True
    else:
        nolegend = False

    if labels_ is None:
        labels_ = ['']

    variable, cut, dataset, labels = ensure_same_length(variable_, cut_, dataset_, labels_)
   
    if type(binning) is AutoBinning or type(binning) is AutoIntCategoryBinning:
        axis = binning.build_auto_axis(variable, cut, dataset)
    elif type(binning) is DefaultBinning:
        axis = binning.build_default_axis(variable[0])
    else:
        axis = binning.build_axis(variable[0])

    fig, ax = setup_canvas()
    add_cms_legend(ax, isdata)

    vals = []
    for v, c, d, l in zip(variable, cut, dataset, labels):
        _, value = plot_histogram_(v, c, d, axis, density, l, ax)
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

    if type(binning) is AutoIntCategoryBinning:
        ticklabels_ints = axis.value(axis.edges[:-1]) # get category labels
        ticklabels_strs = []
        for val in ticklabels_ints:
            if str(val) in binning.label_lookup:
                ticklabels_strs.append(binning.label_lookup[str(val)])
            else:
                ticklabels_strs.append(str(val))

        #major ticks at integer positions
        #no minor ticks
        ax.set_xticks(axis.edges)
        ax.set_xticks([], minor=True)
        ax.set_xticklabels([''] + ticklabels_strs, 
                           rotation=45, ha='right',
                           fontsize=14)
        ax.grid(axis='x', which='major', linestyle='--', alpha=0.7)

    add_text(ax, cut, extratext)

    draw_legend(ax, nolegend)

    plt.tight_layout()

    if output_path is not None:
        savefig(fig, output_path)
        
    plt.close(fig)

def plot_histogram_(variable: AbstractVariable, 
                  cut: AbstractCut, 
                  dataset: AbstractDataset,
                  axis : hist.axis.AxesMixin,
                  density: bool,
                  label: str,
                  ax : plt.Axes):
   
    needed_columns = list(set(variable.columns + cut.columns))
    dataset.ensure_columns(needed_columns)

    cut = cut.evaluate(dataset)
    val = variable.evaluate(dataset)

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