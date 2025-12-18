from .SetupConfig import config
from .Variable import AbstractVariable, variable_from_string, RatioVariable, DifferenceVariable, RelativeResolutionVariable
from .Cut import AbstractCut, common_cuts, NoCut
from .datasets import AbstractDataset
from .Binning import AbstractBinning, AutoBinning, DefaultBinning, AutoIntCategoryBinning

from .histplot import simon_histplot

from .util import setup_canvas, add_cms_legend, savefig, ensure_same_length, add_text, draw_legend, make_oneax, make_axes_withpad

import hist
import matplotlib.pyplot as plt
import matplotlib.axes
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
                   no_ratiopad : bool = False,
                   output_path: Union[str, None] = None):
    
    if labels_ is None or len(labels_) == 1:
        nolegend = True
    else:
        nolegend = False

    if labels_ is None:
        labels_ = ['']

    variable, cut, dataset, labels = ensure_same_length(variable_, cut_, dataset_, labels_)

    if (type(dataset_) is list and (len(dataset_) > 1) or len(dataset) == 1):
        style_from_dset = True
    else:
        style_from_dset = False

    if no_ratiopad or len(variable) == 1:
        do_ratiopad = False
    else:
        do_ratiopad = True

    if type(binning) is AutoBinning or type(binning) is AutoIntCategoryBinning:
        if logx:
            transform='log'
        else:
            transform=None

        axis = binning.build_auto_axis(variable, cut, dataset, transform=transform)
    elif type(binning) is DefaultBinning:
        axis = binning.build_default_axis(variable[0])
    else:
        axis = binning.build_axis(variable[0])

    fig = setup_canvas()

    if do_ratiopad:
        ax_main, ax_pad = make_axes_withpad(fig)
    else:
        ax_main = make_oneax(fig)

    add_cms_legend(ax_main, isdata)

    for v, c, d, l in zip(variable, cut, dataset, labels):
        if style_from_dset and d.label is not None:
            nolegend = False #force legend if dataset has label

        _, _ = d.plot_histogram(
            v, c, axis, 
            density, ax_main, 
            style_from_dset,
            label=l
        )

    if do_ratiopad:
        dnom = dataset[0]

        for v, c, d, l in zip(variable[1:], cut[1:], dataset[1:], labels[1:]):
            d.plot_ratio_to(
                dnom,
                density = density,
                ax = ax_pad, # pyright: ignore[reportPossiblyUnboundVariable]
                own_style = style_from_dset
            )

    if do_ratiopad:
        ax_pad.set_xlabel(variable[0].label) # pyright: ignore[reportPossiblyUnboundVariable]
        if nolegend:
            ax_pad.set_ylabel("Ratio") # pyright: ignore[reportPossiblyUnboundVariable]
        else:
            if style_from_dset:
                denomlabel = dataset[0].label
            else:
                denomlabel = labels[0]
            ax_pad.set_ylabel('Ratio to ' + denomlabel) # pyright: ignore[reportPossiblyUnboundVariable]

    else:
        ax_main.set_xlabel(variable[0].label)

    if density:
        ax_main.set_ylabel('Density')
    else:
        ax_main.set_ylabel('Counts')

    if logx:
        ax_main.set_xscale('log')
    if logy:
        ax_main.set_yscale('log')

    if type(variable[0]) is RatioVariable:
        ax_main.axvline(1.0, color='k', linestyle='dashed')
    elif type(variable[0]) is DifferenceVariable or type(variable[0]) is RelativeResolutionVariable:
        ax_main.axvline(0.0, color='k', linestyle='dashed')

    if do_ratiopad:
        ax_pad.axhline(1.0, color='k', linestyle='dashed') # pyright: ignore[reportPossiblyUnboundVariable]
        ax_pad.grid(axis='y', which='major', linestyle='--', alpha=0.7) # pyright: ignore[reportPossiblyUnboundVariable]

    if type(binning) is AutoIntCategoryBinning:
        # get category label [pylance is confused :(]
        ticklabels_ints = axis.value(axis.edges[:-1]) # pyright: ignore[reportAttributeAccessIssue] 
        ticklabels_strs = []
        for val in ticklabels_ints:
            if str(val) in binning.label_lookup:
                ticklabels_strs.append(binning.label_lookup[str(val)])
            else:
                ticklabels_strs.append(str(val))

        #major ticks at integer positions
        #no minor ticks
        if do_ratiopad:
            ax_pad.set_xticks(axis.edges) # pyright: ignore[reportAttributeAccessIssue, reportPossiblyUnboundVariable]
            ax_pad.set_xticks([], minor=True) # pyright: ignore[reportPossiblyUnboundVariable]
            ax_pad.set_xticklabels([''] + ticklabels_strs,  # pyright: ignore[reportPossiblyUnboundVariable]
                            rotation=45, ha='right', 
                            fontsize=14) 
            ax_pad.grid(axis='x', which='major', linestyle='--', alpha=0.7) # pyright: ignore[reportPossiblyUnboundVariable]
        else:
            ax_main.set_xticks(axis.edges) # pyright: ignore[reportAttributeAccessIssue]
            ax_main.set_xticks([], minor=True)
            ax_main.set_xticklabels([''] + ticklabels_strs, 
                            rotation=45, ha='right',
                            fontsize=14)
            ax_main.grid(axis='x', which='major', linestyle='--', alpha=0.7)

    add_text(ax_main, cut, extratext)

    draw_legend(ax_main, nolegend)

    plt.tight_layout()

    if output_path is not None:
        savefig(fig, output_path)
    else:
        plt.show()
        
    plt.close(fig)