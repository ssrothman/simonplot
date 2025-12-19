from .SetupConfig import config
from .Variable import AbstractVariable, variable_from_string, RatioVariable, DifferenceVariable, RelativeResolutionVariable
from .Cut import AbstractCut, common_cuts, NoCut
from .datasets import AbstractDataset, DatasetStack
from .Binning import AbstractBinning, AutoBinning, DefaultBinning, AutoIntCategoryBinning

from .histplot import simon_histplot

from .util import setup_canvas, add_cms_legend, savefig, ensure_same_length, add_text, draw_legend, make_oneax, make_axes_withpad, get_artist_color

import hist
import matplotlib.pyplot as plt
import matplotlib.axes
import awkward as ak
import numpy as np

from typing import List, Union

def plot_histogram(variable_: Union[AbstractVariable, List[AbstractVariable]], 
                   cut_: Union[AbstractCut, List[AbstractCut]], 
                   weight_ : Union[AbstractVariable, List[AbstractVariable]],
                   dataset_: Union[AbstractDataset, List[AbstractDataset]],
                   binning : AbstractBinning,
                   labels_: Union[List[str], None] = None,
                   extratext : Union[str, None] = None,
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

    variable, cut, weight, dataset, labels = ensure_same_length(variable_, cut_, weight_, dataset_, labels_)

    if (type(dataset_) is list and (len(dataset_) > 1) or len(dataset) == 1):
        style_from_dset = True
        
        first_label = dataset[0].label
        for d in dataset[1:]:
            if d.label == first_label:
                style_from_dset = False
                break
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

    is_stack = np.asarray([isinstance(d, DatasetStack) for d in dataset])
    resolve_stacks = np.sum(is_stack) == 1
    
    if resolve_stacks:
        # make sure to plot resolved stack FIRST
        whichstack = np.where(is_stack)[0][0]

        variable = [variable[whichstack]] + [variable[i] for i in range(len(variable)) if i != whichstack]
        cut = [cut[whichstack]] + [cut[i] for i in range(len(cut)) if i != whichstack]
        weight = [weight[whichstack]] + [weight[i] for i in range(len(weight)) if i != whichstack]
        dataset = [dataset[whichstack]] + [dataset[i] for i in range(len(dataset)) if i != whichstack]
        labels = [labels[whichstack]] + [labels[i] for i in range(len(labels)) if i != whichstack]

    #check if any is data
    is_data = np.asarray([not d.isMC for d in dataset])
    num_data = np.sum(is_data)
    if num_data > 1:
        raise RuntimeError("Cannot plot more than one data dataset")
    elif num_data == 1:
        isdata = True

        which_data = np.where(is_data)[0][0]
        #reorder such that data is LAST
        variable = [variable[i] for i in range(len(variable)) if i != which_data] + [variable[which_data]]
        cut = [cut[i] for i in range(len(cut)) if i != which_data] + [cut[which_data]]
        weight = [weight[i] for i in range(len(weight)) if i != which_data] + [weight[which_data]]
        dataset = [dataset[i] for i in range(len(dataset)) if i != which_data] + [dataset[which_data]]
        labels = [labels[i] for i in range(len(labels)) if i != which_data] + [labels[which_data]]

        which_data = len(variable) - 1
        which_ref = which_data

        for i in range(len(variable)):
            if i == which_data:
                continue

            dataset[i].compute_weight(dataset[which_data].lumi)
    else:
        isdata = False
        which_ref = 0

        for i in range(len(variable)):
            # scale to arbitrary luminosity of 1000fb^{-1}
            dataset[i].compute_weight(1000)

    fig = setup_canvas()

    if do_ratiopad:
        ax_main, ax_pad = make_axes_withpad(fig)
    else:
        ax_main = make_oneax(fig)

    if isdata:
        add_cms_legend(ax_main, True, lumi=dataset[which_data].lumi) # pyright: ignore[reportPossiblyUnboundVariable]
    else:
        add_cms_legend(ax_main, False)

    artists = []
    Hs = []
    for v, c, w, d, l in zip(variable, cut, weight, dataset, labels):
        if style_from_dset and d.label is not None:
            nolegend = False #force legend if dataset has label

        (artist, _) , H = d._plot_histogram(
            v, c, w, axis, 
            density, ax_main, 
            style_from_dset or (not d.isMC),
            label=l,
            fillbetween = 0 if (isinstance(d, DatasetStack) and resolve_stacks) else None
        )
        artists.append(artist)
        Hs.append(H)

    if do_ratiopad:
        Hnom = Hs[which_ref]

        largest_nontrivial_ratio = 1.0
        smallest_nontrivial_ratio = 1.0

        for i, d in enumerate(dataset):
            if i == which_ref:
                continue

            _, ratiovals, ratioerrs = d._plot_ratio(
                Hnom, Hs[i],
                density = density,
                ax = ax_pad, # pyright: ignore[reportPossiblyUnboundVariable]
                own_style = style_from_dset,
                color = get_artist_color(artists[i])
            )

            ratiothreshold = np.nanpercentile(ratioerrs, config['ratiopad']['auto_ylim']['percentile']) 
            ratiothreshold = max(ratiothreshold, config['ratiopad']['auto_ylim']['min_threshold'])

            ratiomask = ratioerrs < ratiothreshold
            largest_nontrivial_ratio = max(
                largest_nontrivial_ratio,
                np.max(ratiovals[ratiomask])
            )

            smallest_nontrivial_ratio = min(
                smallest_nontrivial_ratio,
                np.min(ratiovals[ratiomask])
            )

        pad = config['ratiopad']['auto_ylim']['padding']
        ax_pad.set_ylim( # pyright: ignore[reportPossiblyUnboundVariable]
            smallest_nontrivial_ratio - pad,
            largest_nontrivial_ratio + pad
        )

        ax_pad.set_xlabel(variable[0].label) # pyright: ignore[reportPossiblyUnboundVariable]
        if nolegend:
            ax_pad.set_ylabel("Ratio") # pyright: ignore[reportPossiblyUnboundVariable]
        else:
            if isdata:
                ax_pad.set_ylabel('Data/MC') # pyright: ignore[reportPossiblyUnboundVariable]
            else:
                if style_from_dset:
                    denomlabel = dataset[0].label
                else:
                    denomlabel = labels[0]

                ax_pad.set_ylabel('%s/MC' % denomlabel) # pyright: ignore[reportPossiblyUnboundVariable]

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