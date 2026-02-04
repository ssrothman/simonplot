from simonplot.config import config, check_auto_logx, lookup_axis_label

from simonplot.typing.Protocols import PrebinnedVariableProtocol, VariableProtocol, PrebinnedOperationProtocol, PrebinnedDatasetProtocol, PrebinnedBinningProtocol
from simonplot.util.common import make_radial_ax, prebinned_ylabel, setup_canvas, add_cms_legend, savefig, add_text, draw_legend, make_oneax, make_axes_withpad, get_artist_color, make_fancy_prebinned_labels, label_from_binning

from simonpy.AbitraryBinning import ArbitraryBinning
from simonpy.sanitization import ensure_same_length, all_same_key
from simonpy.text import strip_units, strip_dollar_signs, find_match

import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, LogNorm, SymLogNorm

from typing import List, Union, assert_never
from enum import IntEnum

_ALLOWED_ANGULAR_NAMES = ['phi', 'angle', 'theta', 'c']
_ALLOWED_RADIAL_NAMES = ['r', 'radius']

class _RANGES(IntEnum):
    FULL = 0
    HALF = 1
    QUARTER = 2

def _pcolormesh(ax, edges, angular_name, radial_name, hist2d, cmap, norm):
    return ax.pcolormesh(
        edges[angular_name], edges[radial_name], hist2d,
        shading='auto', rasterized=True,
        cmap=cmap, norm=norm, 
    )


def draw_radial_histogram(
                   variable : PrebinnedVariableProtocol,
                   cut: PrebinnedOperationProtocol, 
                   dataset: PrebinnedDatasetProtocol,
                   binning : PrebinnedBinningProtocol,
                   extratext : Union[str, None] = None,
                   logc : bool | None = None,
                   sym : bool | None = None,
                   output_folder: Union[str, None] = None,
                   output_prefix: Union[str, None] = None):
    
    #get resulting binning
    axis = binning.build_prebinned_axis(dataset, cut)

    if axis.Nax != 2:
        raise ValueError("draw_radial_histogram only supports 2D data")

    if not axis.single_block:
        raise ValueError("draw_radial_histogram only supports single-block (ie rectangular) binning")

    angular_name = find_match(axis.axis_names, _ALLOWED_ANGULAR_NAMES, True)
    if angular_name is None:
        raise ValueError("Could not identify angular axis name from %s (options are %s)" % (axis.axis_names, _ALLOWED_ANGULAR_NAMES))
    
    radial_name = find_match(axis.axis_names, _ALLOWED_RADIAL_NAMES, True)
    if radial_name is None:
        raise ValueError("Could not identify radial axis name from %s (options are %s)" % (axis.axis_names, _ALLOWED_RADIAL_NAMES))
    
    edges = axis.edges
    phirange = np.max(edges[angular_name]) - np.min(edges[angular_name])
    if np.isclose(phirange, 2*np.pi):
        range_type = _RANGES.FULL
    elif np.isclose(phirange, np.pi):
        range_type = _RANGES.HALF
    elif np.isclose(phirange, np.pi/2):
        range_type = _RANGES.QUARTER
    else:
        raise ValueError("Could not determine angular range from edges, got %f" % phirange)
    
    #get values to plot
    hist2d = variable.evaluate(dataset, cut)
    if isinstance(hist2d, tuple):
        hist2d = hist2d[0]
    
    if not isinstance(hist2d, np.ndarray):
        raise ValueError("Variable did not return a numpy ndarray! Instead got %s" % type(hist2d))
    
    extents = [len(edges[name])-1 for name in axis.axis_names]
    hist2d = hist2d.reshape(extents)

    if hist2d.ndim != 2:
        raise ValueError("Variable did not return a 2D histogram! Instead shape was %s" % hist2d.shape)
    
    # attempt to automatically determine logc and sym if not specified
    if logc is None:
        # automatically detect if values are log-like
        pct01 = np.percentile(hist2d, 01.0)
        pct50 = np.percentile(hist2d, 50.0)
        pct99 = np.percentile(hist2d, 99.0)
        if pct01 <= 0:
            logc = False
        else:
            test = (pct99 - pct50) / (pct50 - pct01)
            if test > 10:
                logc = True
            else:
                logc = False

    if sym is None:
        if logc:
            sym = False
        elif variable.centerline is not None:
            sym = True
        else:
            sym = False

    if sym and variable.centerline is None:
        variable.override_centerline(1.0)

    #setup canvas
    fig = setup_canvas()
    ax = make_radial_ax(fig)
    if dataset.isMC:
        add_cms_legend(ax, False)
    else:
        add_cms_legend(ax, True, lumi=dataset.lumi)

    if sym:
        cmap = 'coolwarm'
        diff = hist2d - variable.centerline
        maxabs = np.max(np.abs(diff))
        if logc:
            norm = SymLogNorm(
                linthresh=maxabs/1e3, 
                vmin=variable.centerline-maxabs, 
                vmax=variable.centerline+maxabs
            )
        else:
            norm = Normalize(
                vmin=variable.centerline-maxabs,
                vmax=variable.centerline+maxabs
            )
    else:
        cmap = 'plasma'
        if logc:
            norm = LogNorm()
        else:
            norm = Normalize()

    artist = _pcolormesh(
        ax, edges, angular_name, radial_name, hist2d, cmap, norm
    )

    if range_type == _RANGES.FULL:
        pass
    elif range_type == _RANGES.HALF:
        edges2 = edges.copy()
        edges2[angular_name] = -edges[angular_name]
        _pcolormesh(
            ax, edges2, angular_name, radial_name, hist2d, cmap, norm
        )
    elif range_type == _RANGES.QUARTER:
        edges2 = edges.copy()
        edges2[angular_name] = np.pi - edges[angular_name]
        _pcolormesh(
            ax, edges2, angular_name, radial_name, hist2d, cmap, norm
        )
        edges3 = edges.copy()
        edges3[angular_name] = -edges[angular_name]
        _pcolormesh(
            ax, edges3, angular_name, radial_name, hist2d, cmap, norm
        )
        edges4 = edges.copy()
        edges4[angular_name] = np.pi + edges[angular_name]
        _pcolormesh(
            ax, edges4, angular_name, radial_name, hist2d, cmap, norm
        )
    else:
        assert_never(range_type)

    cbar = fig.colorbar(artist, ax=ax, pad=0.1)

    cbarlabel = prebinned_ylabel(variable, axis)
    cbar.set_label(cbarlabel)

    placed_text = add_text(
        ax, cut, extratext,
        loc=(0.05, 0.95, 'top', 'left')
    )
    
    desired_ticks = [0, 45, 90, 135, 180, 225, 270, 315]
    # the 90 label always clashes with the CMS text
    if '\n' in placed_text: # if text is multiple lines, it will clash with the 135 tick label
        desired_ticklabels = ['0', '45', '', '', '180', '225', '270', '315']
    else: 
        desired_ticklabels = ['0', '45', '', '135', '180', '225', '270', '315']
    
    for i in range(len(desired_ticks)):
        #add degrees symbol to labels
        if desired_ticklabels[i] != '':
            desired_ticklabels[i] = desired_ticklabels[i] + '°'

    ax.set_xticks(np.radians(desired_ticks), labels=desired_ticklabels,)
    #add padding so they don't clash with the actual plot
    ax.tick_params(axis='x', which='major', pad=15)

    fig.tight_layout()

    if output_folder is not None:
        if output_prefix is None:
            output_path = os.path.join(output_folder, 'radial_hist')
        else:
            output_path = os.path.join(output_folder, output_prefix)

        output_path += '_VAR-%s' % variable.key
        output_path += '_CUT-%s' % cut.key
        output_path += '_DSET-%s' % dataset.key
        
        if logc:
            output_path += '_LOGC'

        savefig(fig, output_path)
    else:
        plt.show()

    plt.close(fig)
