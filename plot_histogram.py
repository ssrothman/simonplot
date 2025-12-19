import os
from .SetupConfig import config, check_auto_logx, lookup_axis_label
from .variable.Variable import AbstractVariable, variable_from_string, RatioVariable, DifferenceVariable, RelativeResolutionVariable, PrebinnedVariable
from .cut.Cut import AbstractCut, common_cuts, NoCut
from .datasets import AbstractDataset, UnbinnedDatasetStack
from .binning.Binning import AbstractBinning, AutoBinning, DefaultBinning, AutoIntCategoryBinning
from .binning.PrebinnedBinning import PrebinnedBinning
from .cut.PrebinnedCut import PrebinnedOperation
from .AribtraryBinning import ArbitraryBinning

from .histplot import simon_histplot

from .util import setup_canvas, add_cms_legend, savefig, ensure_same_length, add_text, draw_legend, make_oneax, make_axes_withpad, get_artist_color, all_same_key, strip_units, xlabel_from_binning, make_fancy_prebinned_labels

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
                   logx: Union[bool, None] = None,
                   logy: bool = False,
                   pulls : bool = False,
                   no_ratiopad : bool = False,
                   output_folder: Union[str, None] = None,
                   output_prefix: Union[str, None] = None):

    if labels_ is None or len(labels_) == 1:
        nolegend = True
    else:
        nolegend = False

    if labels_ is None:
        labels_ = ['']

    variable, cut, weight, dataset, labels = ensure_same_length(variable_, cut_, weight_, dataset_, labels_)

    #resolve auto logx BEFORE building axis for unbinned variables
    if logx is None and not isinstance(variable[0], PrebinnedVariable): 
        logx = check_auto_logx(variable[0].key)

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

    if isinstance(binning, AutoBinning) or isinstance(binning, AutoIntCategoryBinning):
        if logx:
            transform='log'
        else:
            transform=None

        axis = binning.build_auto_axis(variable, cut, dataset, transform=transform)
    elif isinstance(binning, DefaultBinning):
        axis = binning.build_default_axis(variable[0])
    elif isinstance(binning, PrebinnedBinning):
        axis = binning.build_prebinned_axis(dataset[0], cut[0])
    else:
        axis = binning.build_axis(variable[0])

    #resolve auto logx AFTER building axis for prebinned variables
    if logx is None and isinstance(axis, ArbitraryBinning):
        if axis.Nax == 1:
            logx = check_auto_logx(axis.axis_names[0])
        else:
            logx = False

    is_stack = np.asarray([isinstance(d, UnbinnedDatasetStack) for d in dataset])
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
        which_data = None

        for i in range(len(variable)):
            # scale to arbitrary luminosity of 1000fb^{-1}
            dataset[i].compute_weight(1000)

    fig = setup_canvas()

    if do_ratiopad:
        ax_main, ax_pad = make_axes_withpad(fig)
    else:
        ax_main = make_oneax(fig)

    if isdata:
        add_cms_legend(ax_main, True, lumi=dataset[which_data].lumi) # pyright: ignore[reportCallIssue, reportArgumentType]
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
            fillbetween = 0 if (isinstance(d, UnbinnedDatasetStack) and resolve_stacks) else None
        )
        artists.append(artist)
        Hs.append(H)

    if do_ratiopad:
        Hnom = Hs[which_ref]

        if pulls:
            largest_nontrivial_ratio = 0.0
            smallest_nontrivial_ratio = 0.0
        else:
            largest_nontrivial_ratio = 1.0
            smallest_nontrivial_ratio = 1.0

        maxthreshold = 0.0

        for i, d in enumerate(dataset):
            if i == which_ref:
                continue

            _, ratiovals, ratioerrs = d._plot_ratio(
                Hnom, Hs[i],
                axis,
                density = density,
                ax = ax_pad, # pyright: ignore[reportPossiblyUnboundVariable]
                own_style = style_from_dset,
                color = get_artist_color(artists[i]),
                pulls = pulls
            )

            ratiothreshold = np.nanpercentile(ratioerrs, config['ratiopad']['auto_ylim']['percentile']) 
            ratiothreshold = max(ratiothreshold, config['ratiopad']['auto_ylim']['min_threshold'])

            ratiomask = ratioerrs < ratiothreshold
            if np.sum(ratiomask) == 0:
                ratiothreshold = np.nanmin(ratioerrs) + 1e-6
                ratiomask = ratioerrs < ratiothreshold

            largest_nontrivial_ratio = max(
                largest_nontrivial_ratio,
                np.nanmax(ratiovals[ratiomask])
            )

            smallest_nontrivial_ratio = min(
                smallest_nontrivial_ratio,
                np.nanmin(ratiovals[ratiomask])
            )

            maxthreshold = max(maxthreshold, ratiothreshold)

        pad = config['ratiopad']['auto_ylim']['padding'] + maxthreshold
        ax_pad.set_ylim( # pyright: ignore[reportPossiblyUnboundVariable]
            smallest_nontrivial_ratio - pad,
            largest_nontrivial_ratio + pad
        )

        if isinstance(axis, ArbitraryBinning):
            ax_pad.set_xlabel(xlabel_from_binning(axis)) # pyright: ignore[reportPossiblyUnboundVariable]
        else:
            ax_pad.set_xlabel(variable[0].label) # pyright: ignore[reportPossiblyUnboundVariable]

        if pulls:
            extra_ylabel = ' [pull]'
        else:
            extra_ylabel = ''

        if nolegend:
            ax_pad.set_ylabel("Ratio" + extra_ylabel) # pyright: ignore[reportPossiblyUnboundVariable]
        else:
            if isdata:
                ax_pad.set_ylabel('Data/MC' + extra_ylabel) # pyright: ignore[reportPossiblyUnboundVariable]
            else:
                if style_from_dset:
                    denomlabel = dataset[0].label
                else:
                    denomlabel = labels[0]

                ax_pad.set_ylabel(('%s/MC' % denomlabel) + extra_ylabel) # pyright: ignore[reportPossiblyUnboundVariable]

    else:
        if isinstance(axis, ArbitraryBinning):
            ax_main.set_xlabel(xlabel_from_binning(axis)) 
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
        if pulls:
            ax_pad.axhline(0.0, color='k', linestyle='dashed') # pyright: ignore[reportPossiblyUnboundVariable]
            xmin, xmax = ax_pad.get_xlim() # pyright: ignore[reportPossiblyUnboundVariable]
            ax_pad.fill_between([xmin, xmax], -1.0, 1.0, color='gray', alpha=0.3) # pyright: ignore[reportPossiblyUnboundVariable]
            ax_pad.set_xlim(xmin, xmax) # pyright: ignore[reportPossiblyUnboundVariable]
        else:
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
            ax_main.set_xticks(axis.edges) # pyright: ignore[reportArgumentType, reportAttributeAccessIssue]
            ax_main.set_xticks([], minor=True)
            ax_main.set_xticklabels([''] + ticklabels_strs, 
                            rotation=45, ha='right',
                            fontsize=14)
            ax_main.grid(axis='x', which='major', linestyle='--', alpha=0.7)
    
    elif type(axis) is ArbitraryBinning:
        make_fancy_prebinned_labels(
            ax_main,
            ax_pad if do_ratiopad else None, # pyright: ignore[reportPossiblyUnboundVariable]
            axis
        )

    add_text(ax_main, cut, extratext)

    draw_legend(ax_main, nolegend)

    plt.tight_layout()

    if output_folder is not None:
        if output_prefix is None:
            output_path = os.path.join(output_folder, 'hist')
        else:
            output_path = os.path.join(output_folder, output_prefix)

        if all_same_key(variable):
            output_path += '_VAR-%s' % variable[0].key 
        
        if all_same_key(cut):
            output_path += '_CUT-%s' % cut[0].key

        if all_same_key(weight, skip=which_data):
            output_path += '_WGT-%s' % weight[0].key

        if all_same_key(dataset):
            output_path += '_DSET-%s' % dataset[0].key
        elif all_same_key(dataset, skip=which_data):
            output_path += '_DSET-DATAvs%s' % dataset[0].key
        else:
            output_path += '_DSET-%s' % ('vs'.join([d.key for d in dataset]))
        if logx:
            output_path += '_LOGX'
        if logy:
            output_path += '_LOGY'
        if density:
            output_path += '_DENSITY'
        if no_ratiopad:
            output_path += '_NORATIO'

        if pulls and do_ratiopad:
            output_path += '_PULLS'

        savefig(fig, output_path)
    else:
        plt.show()
        
    plt.close(fig)