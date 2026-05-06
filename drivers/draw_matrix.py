import os
from typing import Union, List, Literal, get_args

import matplotlib
from matplotlib.colors import Normalize, SymLogNorm, LogNorm

from simonplot.config.lookuputil import lookup_axis_label
from simonplot.variable.PrebinnedVariable import _ExtractCovarianceMatrix
from simonplot.typing.Protocols import PrebinnedOperationProtocol, PrebinnedDatasetProtocol, PrebinnedBinningProtocol, PrebinnedVariableProtocol, VariableProtocol
from simonplot.util.common import add_axis_label, add_text, label_from_binning, make_fancy_prebinned_labels, setup_canvas, make_oneax, savefig, add_cms_legend
from simonpy.AbitraryBinning import ArbitraryBinning, ArbitraryGenRecoBinning
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
                output_prefix: Union[str, None] = None,
                override_filename: Union[str, None] = None,
                override_cbarlabel: Union[str, None] = None):
    
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


    if isinstance(axis, ArbitraryBinning):
        if axis.Nax == 1:
            xedges = axis.edges[axis.axis_names[0]]
            yedges = xedges
        else:
            xedges = None
            yedges = None
    elif isinstance(axis, ArbitraryGenRecoBinning):
        if axis.recobinning.Nax == 1:
            yedges = axis.recobinning.edges[axis.recobinning.axis_names[0]]
        else:
            yedges = None
        
        if axis.genbinning.Nax == 1:
            xedges = axis.genbinning.edges[axis.genbinning.axis_names[0]]
        else:
            xedges = None
    else:
        raise ValueError("Unsupported axis type: %s" % type(axis))


    #need to clip +-inf
    if xedges is not None:
        if xedges[0] == -np.inf:
            print("WARNING: clipping underflow bin on x axis")
            xedges = xedges[1:]
            mat = mat[1:, :]
        if xedges is not None and xedges[-1] == +np.inf:
            print("WARNING: clipping overflow bin on x axis")
            xedges = xedges[:-1]
            mat = mat[:-1, :]

        # attempt to detect logarithmic binning
        # if bin spacing between first two bins is much smaller than 
        # spacing between last two bins, assume log binning
        lower = xedges[1] - xedges[0]
        upper = xedges[-1] - xedges[-2]
        if upper / lower > 2:
            ax.set_xscale('log')
    else:
        xedges = np.arange(mat.shape[1] + 1) - 0.5

    if yedges is not None:
        if yedges[0] == -np.inf:
            print("WARNING: clipping underflow bin on y axis")
            yedges = yedges[1:]
            mat = mat[:, 1:]
        if yedges is not None and yedges[-1] == +np.inf:
            print("WARNING: clipping overflow bin on y axis")
            yedges = yedges[:-1]
            mat = mat[:, :-1]

        # attempt to detect logarithmic binning
        # if bin spacing between first two bins is much smaller than 
        # spacing between last two bins, assume log binning
        lower = yedges[1] - yedges[0]
        upper = yedges[-1] - yedges[-2]
        if upper / lower > 2:
            ax.set_yscale('log')
    else:
        yedges = np.arange(mat.shape[0] + 1) - 0.5


    artist = ax.pcolormesh(xedges, yedges, mat, cmap=cmap, norm=normobj, rasterized=True)

    if isinstance(axis, ArbitraryGenRecoBinning):
        the_ylabel = label_from_binning(axis.recobinning)
        the_xlabel = label_from_binning(axis.genbinning)
        add_axis_label(ax, the_xlabel, which='x')
        add_axis_label(ax, the_ylabel, which='y')
    elif isinstance(axis, ArbitraryBinning):
        the_xlabel = label_from_binning(axis)
        add_axis_label(ax, the_xlabel, which='x')
        add_axis_label(ax, the_xlabel, which='y')
    else:
        raise ValueError("Unsupported axis type: %s" % type(axis))

    cbar = fig.colorbar(artist, ax=ax)
    
    if override_cbarlabel is not None:
        cbar.set_label(override_cbarlabel)
    else:
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
    
    if isinstance(axis, ArbitraryBinning):
        fontsize_offset, fallback_rotation = make_fancy_prebinned_labels(ax, axis, 'x')
        make_fancy_prebinned_labels(ax, axis, 'y',
                                    fontsize_offset = fontsize_offset,
                                    fallback_rotation=fallback_rotation)
    elif isinstance(axis, ArbitraryGenRecoBinning):
        fontsize_offset, fallback_rotation = make_fancy_prebinned_labels(ax, axis.genbinning, 'x')
        make_fancy_prebinned_labels(ax, axis.recobinning, 'y',
                                    fontsize_offset = fontsize_offset,
                                    fallback_rotation=fallback_rotation)
    else:
        raise ValueError("Unsupported axis type: %s" % type(axis))

    ax.set_box_aspect(1)

    fig.tight_layout()

    if output_folder is not None:
        if override_filename is not None:
            output_path = os.path.join(output_folder, override_filename)
        else:
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