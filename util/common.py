import matplotlib
import matplotlib.figure
import matplotlib.axes
import matplotlib.lines
import matplotlib.container
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep

from simonplot.config import config, lookup_axis_label
from .place_text import place_text

from simonplot.cut import common_cuts, NoCut
from simonplot.typing.Protocols import CutProtocol, PrebinnedVariableProtocol

from simonpy.AbitraryBinning import ArbitraryBinning
from simonpy.text import clean_string, strip_units

from typing import Literal, Tuple, Union, List, reveal_type

hep.style.use(hep.style.CMS)
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['axes.formatter.useoffset'] = False
matplotlib.rcParams['axes.formatter.use_mathtext'] = True


def setup_canvas() -> matplotlib.figure.Figure:
    fig = plt.figure(figsize=config['figsize'])

    return fig

def make_oneax(fig: matplotlib.figure.Figure) -> matplotlib.axes.Axes:
    ax = fig.add_subplot(1,1,1)
    return ax

def make_radial_ax(fig: matplotlib.figure.Figure) -> matplotlib.axes.Axes:
    ax = fig.add_subplot(1,1,1, projection='polar')
    return ax

def make_axes_withpad(fig: matplotlib.figure.Figure):
    (ax_main, ax_pad) = fig.subplots(
        2, 1,
        gridspec_kw={'height_ratios': [1, config['ratiopad']['height']],
                     'hspace': config['ratiopad']['hspace']},
        sharex=True
    )
    return (ax_main, ax_pad)

def add_cms_legend(ax, isdata: bool, lumi: Union[float, None]=None):
    if isdata:
        hep.cms.label(ax=ax, data=True, 
                      lumi='%0.2f'%lumi,
                      year= config.get('year', None),
                      com = config.get('com', None),
                      label=config['cms_label'])
    else:
        hep.cms.label(ax=ax, data=False, 
                      label=config['cms_label'])
        
def savefig(fig, path: str, mkdir : bool=True):
    if mkdir:
        import os
        dirname = os.path.dirname(path)
        if dirname != '' and dirname != path:
            os.makedirs(dirname, exist_ok=True)

    print("Saving figure %s" % (path))
    fig.savefig(path+'.png', dpi=300, bbox_inches='tight', format='png')
    fig.savefig(path+'.pdf', bbox_inches='tight', format='pdf')

def check_ticklabel_overlap(ax : matplotlib.axes.Axes) -> bool:
        fig = ax.get_figure()
        if fig is None:
            raise RuntimeError("make_fancy_prebinned_labels: could not get figure from axis")
        
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer() # pyright: ignore[reportAttributeAccessIssue]

        labels = ax.get_xticklabels()
        bboxes = [label.get_window_extent(renderer) for label in labels]
        bboxes = [bbox.expanded(1.15, 1.15) for bbox in bboxes]

        for i in range(len(bboxes)-1):
            if bboxes[i].overlaps(bboxes[i+1]):
                return True
            
        return False

def make_fancy_prebinned_labels(ax : matplotlib.axes.Axes, 
                                axis : ArbitraryBinning,
                                which : Literal['x', 'y']= 'x',
                                skip_labels : bool = False,
                                fontsize_offset : int = 0,
                                fallback_rotation : float = 0):
    
    original_xlim = ax.get_xlim()
    original_ylim = ax.get_ylim()

    cfg = config['fancy_prebinned_labels']
    if not cfg['enabled']:
        return fontsize_offset, fallback_rotation #short circuit if not enabled
    
    if axis.Nax > cfg['max_ndim'] or axis.Nax == 1:
        return fontsize_offset, fallback_rotation #short circuit if too many dimensions, or only 1 dimension (in which case fancy labels don't make sense)

    blocks = axis.get_blocks([axis.axis_names[0]])
    
    all_contiguous = True
    for block in blocks:
        if not isinstance(block['slice'], slice):
            all_contiguous = False
            break
    
    if not all_contiguous:
        print("WARNING: fancy prebinned labels only supported for contiguous blocks along axis")
        return fontsize_offset, fallback_rotation
    
    major_ticks = []
    for block in blocks:
        start = block['slice'].start 
        major_ticks.append(start)
    major_ticks.append(axis.total_size)
    major_ticks = np.asarray(major_ticks)

    if which == 'x':
        set_ticks_fun = ax.set_xticks
    else:
        set_ticks_fun = ax.set_yticks

    set_ticks_fun(major_ticks, minor=False, labels=['']*len(major_ticks))
    
    if axis.total_size < cfg['max_minor_ticks']:
        set_ticks_fun(np.arange(axis.total_size + 1) - 0.5, minor=True)

    ax.grid(axis=which, which='major', linestyle='--', alpha=0.9)
        
    if skip_labels:
        return fontsize_offset, fallback_rotation # no more work needed
    
    #twin axis so that we can put labels BETWEEN the major ticks
    #by making invisible major ticks in the correct positions
    if which == 'x':
        ax2 = ax.twiny()# pyright: ignore[reportPossiblyUnboundVariable]
    else:
        ax2 = ax.twinx()# pyright: ignore[reportPossiblyUnboundVariable]

    ax.tick_params(direction='inout', which='major', 
                    axis=which, length=cfg['ticksize']) 

    #hide all the spines
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    #put the relevant spines in the correct place
    #by default they are opposite the original axes,
    #but we want them on top of the original axes
    if which == 'x':
        ax2.spines['top'].set_position(('axes', 0.0))
    else:
        ax2.spines['right'].set_position(('axes', 0.0))

    #compute tick positions and labels
    major_tick_centers = (major_ticks[:-1] + major_ticks[1:]) / 2
    major_tick_labels = []
    axname = lookup_axis_label(axis.axis_names[0])
    axname = strip_units(axname)
    for block in blocks:
        low = block['edges'][axis.axis_names[0]][0]
        high = block['edges'][axis.axis_names[0]][1]
        if low == -np.inf:
            major_tick_labels.append('%s$ \\leq %0.3g$' % (axname, high))
        elif high == np.inf:
            major_tick_labels.append('$%0.3g < $%s' % (low, axname))
        else:
            major_tick_labels.append('$%0.3g < $%s$ \\leq %0.3g$' % (low, axname, high))

    if which == 'x':
        set_ticks_fun = ax2.set_xticks
        extra_set_ticks_params = {}
        extra_tick_params = {'labelrotation' : fallback_rotation}
    else:
        set_ticks_fun = ax2.set_yticks
        extra_set_ticks_params = {'va' : 'center'}
        extra_tick_params = {'labelrotation' : 90 - fallback_rotation}

    set_ticks_fun(
        major_tick_centers, minor=False,
        labels=major_tick_labels, 
        **extra_set_ticks_params
    ) 
    set_ticks_fun([], minor=True)
    whichlabel = {
        'labelbottom' : False,
        'labeltop' : False,
        'labelleft' : False,
        'labelright' : False
    }
    if which == 'x':
        whichlabel['labelbottom'] = True
    else:
        whichlabel['labelleft'] = True

    ax2.tick_params(
        axis=which, direction='in', which='both',
        **whichlabel,
        labelsize=cfg['fontsize'] - fontsize_offset,
        length=0,
        **extra_tick_params
    )

    #feature is optional because it might be slow
    if cfg['check_label_overlap'] and which == 'x':
        while (check_ticklabel_overlap(ax2) and cfg['fontsize'] - fontsize_offset > cfg['min_fontsize']):
            fontsize_offset += 1
            ax2.tick_params(axis=which, labelsize=cfg['fontsize'] - fontsize_offset)
        if fontsize_offset > 0 and cfg['fontsize'] - fontsize_offset <= cfg['min_fontsize']:
            print("WARNING: fancy prebinned labels had overlapping label even at minimum fontsize %d"%(cfg['min_fontsize']))
            print("\tRotating axis labels by 30 degrees")
            fallback_rotation = 30.
            ax2.tick_params(axis=which, labelrotation=fallback_rotation)
        elif fontsize_offset > 0 and cfg['fontsize'] - fontsize_offset > cfg['min_fontsize']:
            print("WARNING: fancy prebinned labels had overlapping labels, reduced fontsize to %d"%(cfg['fontsize'] - fontsize_offset))

    #ensure we didn't break the original axis limits
    #and that the twin axis matches the original axis
    ax.set_xlim(original_xlim)
    ax2.set_xlim(original_xlim)
    
    ax.set_ylim(original_ylim)
    ax2.set_ylim(original_ylim)

    return fontsize_offset, fallback_rotation

def add_axis_label(ax : matplotlib.axes.Axes, label : str, which : Literal['x', 'y']):
    default_fontsize = config['%slabel_fontsize'%(which)]
    fontsize_offset = 0
    
    
    if which == 'y':
        setlabel_func = ax.set_ylabel
    else:
        setlabel_func = ax.set_xlabel

    setlabel_func(label, fontsize=default_fontsize)

    #get label extent on the plot
    label_extent = ax.yaxis.get_label().get_window_extent(renderer=ax.figure.canvas.get_renderer()) # pyright: ignore[reportAttributeAccessIssue]
    # transform label extent into units where 0 = top of axis, 1 = bottom of axis
    label_extent = label_extent.transformed(ax.transAxes.inverted())
    while (which == 'y' and label_extent.y0 < 0.0):
        fontsize_offset = fontsize_offset + 1
        setlabel_func(label, fontsize=default_fontsize - fontsize_offset)
        label_extent = ax.yaxis.get_label().get_window_extent(renderer=ax.figure.canvas.get_renderer()) # pyright: ignore[reportAttributeAccessIssue]
        label_extent = label_extent.transformed(ax.transAxes.inverted())

    if fontsize_offset > 0:
        print("Warning: %s-axis label font size had to be reduced by %d points to fit into axis!" % (which, fontsize_offset))

    return fontsize_offset

def draw_legend(ax: matplotlib.axes.Axes, nolegend: bool, scale: float=1.0, loc: Union[str, int, tuple] ='best'):
    if not nolegend:
        if type(loc) in [int, str]:
            ldg = ax.legend(
                fontsize=18, 
                loc=loc,
                framealpha=0.8,
                borderpad=0.3,
                frameon=True,
                markerscale=scale
            )
        elif type(loc) is tuple:
            ldg = ax.legend(
                fontsize=18, 
                bbox_to_anchor=loc[:2],
                loc=loc[2],
                framealpha=0.8,
                borderpad=0.3,
                frameon=True,
                markerscale=scale
            )
        else:
            raise ValueError("draw_legend: loc argument has invalid type %s"%(type(loc)))

        #if markers are tiny, increase their size in legend
        for handle in ldg.legend_handles: # pyright: ignore[reportAttributeAccessIssue]
            if hasattr(handle, 'get_sizes'):
                
                minsize = np.min(handle.get_sizes()) # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]

                if minsize < 25: #arbitrary threshold. NB this is the square of the markersize

                    #remove existing legend, and recursively callback with a larger markerscale
                    ldg.remove()
                    draw_legend(ax, nolegend, scale=scale+1, loc=loc)
                    break

def add_text(ax : matplotlib.axes.Axes, 
             cut: Union[CutProtocol, List[CutProtocol]], 
             extratext: Union[str, None]=None,
             loc : str | int | Tuple[float, float, str, str]='best'):
    ccut = common_cuts(cut)
    if not isinstance(cut, NoCut):
        thetext = ccut.label
    else:
        thetext = ''
        
    if extratext is not None:
        thetext = extratext + '\n' + thetext

    thetext = thetext.strip()

    if thetext != '':
        place_text(ax, thetext, loc=loc, fontsize=24, bbox_opts={
            'boxstyle': 'round,pad=0.3',
            'facecolor': 'white',
            'edgecolor': 'black',
            'alpha': 0.8
        })
    return thetext

def get_artist_color(artist : Union[matplotlib.container.ErrorbarContainer, matplotlib.patches.Patch, matplotlib.lines.Line2D]):
    if isinstance(artist, matplotlib.patches.Patch):
        return artist.get_facecolor()
    elif isinstance(artist, matplotlib.container.ErrorbarContainer):
        return get_artist_color(artist.lines[0])
    elif isinstance(artist, matplotlib.lines.Line2D):
        return artist.get_color()

def label_from_binning(binning : ArbitraryBinning) -> str:
    if binning.Nax == 1:
        return lookup_axis_label(binning.axis_names[0])
    else:
        return '@'.join([strip_units(lookup_axis_label(ax)) for ax in binning.axis_names]) + " bin index"
    

def prebinned_ylabel(var : PrebinnedVariableProtocol, binning : ArbitraryBinning) -> str:
    if var.normalized_by_err:
        ylabel = '$\\frac{N}{\\sigma_N}$'
    elif var.hasjacobian:
        denom = ''
        axes = var.jac_details['wrt']
        if len(axes) == 0:
            axes = binning.axis_names
        for ax in axes:
            l = clean_string(lookup_axis_label(ax))
            if ax in var.jac_details['radial_coords']:
                denom += l + 'd(' + l + ')'
            else:
                denom += 'd(' + l + ')'
        ylabel = '$\\frac{dN}{%s}$' % denom
    else:
        ylabel = 'Bin counts'

    if var.normalized_blocks:
        normvars = var.normalized_blocks

        normvars_for_extra = []
        for normvar in normvars:
            if normvar not in var.jac_details['wrt']:
                normvars_for_extra.append(normvar)
            else:
                l = clean_string(lookup_axis_label(normvar))
                ylabel = ylabel.replace(l + 'd(%s)' % l, '')
                ylabel = ylabel.replace('d(%s)'%l, '')

        if normvars_for_extra:
            if len(normvars_for_extra) == 1:
                binsid = strip_units(lookup_axis_label(normvars_for_extra[0]))
            else:
                binsid = '(%s)' % ', '.join([strip_units(lookup_axis_label(vv)) for vv in normvars_for_extra])

            ylabel += ' (normalized per %s bin)' % binsid

    return ylabel