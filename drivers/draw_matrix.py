import os
from typing import Union, List, Literal, get_args

import matplotlib
from matplotlib.colors import Normalize, SymLogNorm, LogNorm

from simonplot.config.lookuputil import lookup_axis_label
from simonplot.variable.PrebinnedVariable import _ExtractCovarianceMatrix
from simonplot.typing.Protocols import PrebinnedOperationProtocol, PrebinnedDatasetProtocol, PrebinnedBinningProtocol, PrebinnedVariableProtocol, VariableProtocol
from simonplot.util.common import add_axis_label, add_text, label_from_binning, make_fancy_prebinned_labels, setup_canvas, make_oneax, savefig, add_cms_legend
from simonpy.text import strip_units

import numpy as np
import matplotlib.pyplot as plt

def draw_matrix(variable : PrebinnedVariableProtocol,
                cut: PrebinnedOperationProtocol, 
                dataset: PrebinnedDatasetProtocol,
                binning : PrebinnedBinningProtocol,
                extratext : Union[str, None] = None,
                sym : Union[bool, None] = None,
                logc : bool = False,
                output_folder: Union[str, None] = None,
                output_prefix: Union[str, None] = None):
    
    variable = _ExtractCovarianceMatrix(variable)

    #get resulting binning
    axis = binning.build_prebinned_axis(dataset, cut)

    #get matrix to plot
    mat = variable.evaluate(dataset, cut)
    if not isinstance(mat, np.ndarray):
        raise ValueError("Variable did not return a numpy ndarray! Instead got %s" % type(mat))
    if mat.ndim != 2:
        raise ValueError("Variable did not return a 2D matrix! Instead shape was %s" % mat.shape)

    #automatically determine if values are (conceptually) symmetric about zero
    if sym is None:
        if 'CorrelationFromCovariance' in variable.key:
            sym = True
        elif np.all(mat >= 0):
            sym = False
        else:
            test = np.min(mat) / np.max(mat)
            if test < -0.5 and test > -2:
                sym = True
            else:
                sym = False
        

    if sym:
        cmap = 'coolwarm'
        extreme = np.max(np.abs(mat))
        if logc:
            normobj = SymLogNorm(
                vmin = -extreme,
                vmax = extreme,
                linthresh=extreme/1e3,
                linscale=1e-1,
            )
        else:
            normobj = Normalize(
                vmin = -extreme,
                vmax = extreme,
            )
    else:
        cmap = 'plasma'
        if logc:
            normobj = LogNorm()
        else:
            normobj = Normalize()
        
    
    fig = setup_canvas()
    ax = make_oneax(fig)
    if dataset.isMC:
        add_cms_legend(ax, False)
    else:
        add_cms_legend(ax, True, lumi=dataset.lumi)

    if axis.Nax == 1: # axis edges cam be physical values!
        edges = axis.edges[axis.axis_names[0]]

        #need to clip +-inf
        if edges[0] == -np.inf:
            print("WARNING: clipping underflow bin")
            edges = edges[1:]
            mat = mat[1:, 1:]
        if edges[-1] == +np.inf:
            print("WARNING: clipping overflow bin")
            edges = edges[:-1]
            mat = mat[:-1, :-1]

        artist = ax.pcolormesh(edges, edges, mat, cmap=cmap, norm=normobj, rasterized=True)

        # attempt to detect logarithmic binning
        # if bin spacing between first two bins is much smaller than 
        # spacing between last two bins, assume log binning
        lower = edges[1] - edges[0]
        upper = edges[-1] - edges[-2]
        if upper / lower > 2:
            ax.set_xscale('log')
            ax.set_yscale('log')
    else:
        artist = ax.pcolormesh(mat, cmap=cmap, norm=normobj, rasterized=True)

    the_xlabel = label_from_binning(axis)
    add_axis_label(ax, the_xlabel, which='x')
    add_axis_label(ax, the_xlabel, which='y')

    cbar = fig.colorbar(artist, ax=ax)
    
    if variable.normalized_by_err:
        cbarlabel = 'Correlation'
    else:
        if variable.hasjacobian:
            cbarlabel = 'Covariance density'
        else:
            cbarlabel = 'Covariance'

    if variable.normalized_blocks:
        normvars = variable.normalized_blocks
        if len(normvars) == 1:
            binsid = strip_units(lookup_axis_label(normvars[0]))
        else:
            binsid = '(%s)' % ', '.join([strip_units(lookup_axis_label(v)) for v in normvars])

        cbarlabel += ' (normalized per %s bin)' % binsid

    cbar.set_label(cbarlabel)

    add_text(ax, cut, extratext)
    
    fontsize_offset, fallback_rotation = make_fancy_prebinned_labels(ax, axis, 'x')
    make_fancy_prebinned_labels(ax, axis, 'y',
                                fontsize_offset = fontsize_offset,
                                fallback_rotation=fallback_rotation)

    ax.set_box_aspect(1)

    fig.tight_layout()

    if output_folder is not None:
        if output_prefix is None:
            output_path = os.path.join(output_folder, 'matrix')
        else:
            output_path = os.path.join(output_folder, output_prefix)

        output_path += '_VAR-%s' % variable.key
        output_path += '_CUT-%s' % cut.key
        output_path += '_DSET-%s' % dataset.key
        
        if logc:
            output_path += '_LOGC'
        if sym:
            output_path += '_SYM'

        savefig(fig, output_path)
    else:
        plt.show()

    plt.close(fig)