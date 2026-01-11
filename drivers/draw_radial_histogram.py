from simonplot.config import config, check_auto_logx, lookup_axis_label

from simonplot.typing.Protocols import PrebinnedVariableProtocol, VariableProtocol, PrebinnedOperationProtocol, PrebinnedDatasetProtocol, PrebinnedBinningProtocol
from simonplot.util.common import make_radial_ax, setup_canvas, add_cms_legend, savefig, add_text, draw_legend, make_oneax, make_axes_withpad, get_artist_color, make_fancy_prebinned_labels, label_from_binning

from simonpy.AbitraryBinning import ArbitraryBinning
from simonpy.sanitization import ensure_same_length, all_same_key
from simonpy.text import strip_units, strip_dollar_signs, find_match

import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, LogNorm, SymLogNorm

from typing import List, Union
from enum import IntEnum

_ALLOWED_ANGULAR_NAMES = ['phi', 'angle', 'theta', 'c']
_ALLOWED_RADIAL_NAMES = ['r', 'radius']

class _RANGES(IntEnum):
    FULL = 0
    HALF = 1
    QUARTER = 2

def draw_radial_histogram(
                   variable : PrebinnedVariableProtocol,
                   cut: PrebinnedOperationProtocol, 
                   dataset: PrebinnedDatasetProtocol,
                   binning : PrebinnedBinningProtocol,
                   extratext : Union[str, None] = None,
                   logc : bool = True,
                   output_folder: Union[str, None] = None,
                   output_prefix: Union[str, None] = None):
    
    #get resulting binning
    axis = binning.build_prebinned_axis(dataset, cut)

    if axis.Nax != 2:
        raise ValueError("draw_radial_histogram only supports 2D data")

    if not axis.single_block:
        raise ValueError("draw_radial_histogram only supports single-block (ie rectangular) binning")

    radial_name = find_match(axis.axis_names, _ALLOWED_ANGULAR_NAMES, True)
    if radial_name is None:
        raise ValueError("Could not identify angular axis name from %s (options are %s)" % (axis.axis_names, _ALLOWED_ANGULAR_NAMES))
    
    angular_name = find_match(axis.axis_names, _ALLOWED_RADIAL_NAMES, True)
    if angular_name is None:
        raise ValueError("Could not identify radial axis name from %s (options are %s)" % (axis.axis_names, _ALLOWED_RADIAL_NAMES))
    
    edges = axis.edges
    phirange = np.max(edges[radial_name]) - np.min(edges[radial_name])
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
    if not isinstance(hist2d, np.ndarray):
        raise ValueError("Variable did not return a numpy ndarray! Instead got %s" % type(hist2d))
    if hist2d.ndim != 2:
        raise ValueError("Variable did not return a 2D histogram! Instead shape was %s" % hist2d.shape)

    #setup canvas
    fig = setup_canvas()
    ax = make_radial_ax(fig)
    if dataset.isMC:
        add_cms_legend(ax, False)
    else:
        add_cms_legend(ax, True, lumi=dataset.lumi)

    cmap = 'viridis'
    if logc:
        norm = LogNorm()
    else:
        norm = Normalize()

    artist = ax.pcolormesh(
        edges[radial_name], edges[angular_name], hist2d,
        shading='auto', rasterized=True,
        cmap=cmap, norm=norm, 
    )

    cbar = fig.colorbar(artist, ax=ax, pad=0.1)

    if variable.normalized_by_err:
        cbarlabel = '$\\frac{N}{\\sigma_N}$'
    elif variable.hasjacobian:
        radial_label = strip_dollar_signs(strip_units(lookup_axis_label(radial_name)))
        angular_label = strip_dollar_signs(strip_units(lookup_axis_label(angular_name)))
        cbarlabel = '$\\frac{dN}{%sd%sd%s}$'% (radial_label, radial_label, angular_label)
    else:
        cbarlabel = 'Bin counts'

    cbar.set_label(cbarlabel)

    add_text(ax, cut, extratext)

    fig.tight_layout()

    if output_folder is not None:
        if output_prefix is None:
            output_path = os.path.join(output_folder, 'radial_hist')
        else:
            output_path = os.path.join(output_folder, output_prefix)

        output_path += '_CUT-%s' % cut.key
        output_path += '_DSET-%s' % dataset.key

        output_path += '_TO-DO'
        
        if logc:
            output_path += '_LOGC'

        savefig(fig, output_path)
    else:
        plt.show()

    plt.close(fig)
