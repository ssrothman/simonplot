import matplotlib
import matplotlib.figure
import matplotlib.axes
import matplotlib.lines
import matplotlib.container
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep

from .config import config, lookup_axis_label
from .place_text import place_text

from simon_mpl_util.plotting.cut.Cut import AbstractCut, common_cuts, NoCut

from simon_mpl_util.util.AribtraryBinning import ArbitraryBinning
from simon_mpl_util.util.text import strip_units

from typing import Union, List

hep.style.use(hep.style.CMS)
matplotlib.rcParams['savefig.dpi'] = 300

def setup_canvas() -> matplotlib.figure.Figure:
    fig = plt.figure(figsize=config['figsize'])

    return fig

def make_oneax(fig: matplotlib.figure.Figure) -> matplotlib.axes.Axes:
    ax = fig.add_subplot(1,1,1)
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

def make_fancy_prebinned_labels(ax_main : matplotlib.axes.Axes, 
                                ax_pad : Union[matplotlib.axes.Axes, None],
                                axis : ArbitraryBinning):
    original_xlim = ax_main.get_xlim()

    cfg = config['fancy_prebinned_labels']
    if not cfg['enabled']:
        return
    
    if axis.Nax > cfg['max_ndim'] or axis.Nax == 1:
        return

    blocks = axis.get_blocks([axis.axis_names[0]])
    
    all_contiguous = True
    for block in blocks:
        if not isinstance(block['slice'], slice):
            all_contiguous = False
            break
    
    if not all_contiguous:
        print("WARNING: fancy prebinned labels only supported for contiguous blocks along axis")
        return
    
    major_ticks = []
    for block in blocks:
        start = block['slice'].start 
        major_ticks.append(start - 0.5)
    major_ticks.append(axis.total_size - 0.5)
    major_ticks = np.asarray(major_ticks)
    ax_main.set_xticks(major_ticks, minor=False, labels=['']*len(major_ticks))
    ax_main.set_xticks(np.arange(axis.total_size + 1) - 0.5, minor=True)
    ax_main.grid(axis='x', which='major', linestyle='--', alpha=0.9)

    if ax_pad is not None:
        ax_pad.grid(axis='x', which='major', linestyle='--', alpha=0.9) # pyright: ignore[reportPossiblyUnboundVariable]
        bottom_ax = ax_pad # pyright: ignore[reportPossiblyUnboundVariable]
        
        ax2 = ax_pad.twiny()# pyright: ignore[reportPossiblyUnboundVariable]
    else:
        ax2 = ax_main.twiny()
        bottom_ax = ax_main

    bottom_ax.tick_params(direction='inout', which='major', 
                            axis='x', length=cfg['ticksize']) 

    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_position(('axes', 0.0))
    ax2.set_xlim(ax_main.get_xlim())
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

    ax2.set_xticks(major_tick_centers, minor=False,
                    labels=major_tick_labels) #dummy labels
    ax2.set_xticks([], minor=True)
    ax2.tick_params(axis='x', direction='in', which='both',
                    labelbottom=True, labeltop=False,
                    labelsize=cfg['fontsize'],
                    length=0)
    
    ax_main.set_xlim(original_xlim)
    ax2.set_xlim(original_xlim)
    return ax2

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

def add_text(ax : matplotlib.axes.Axes, cut: Union[AbstractCut, List[AbstractCut]], extratext: Union[str, None]=None):
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