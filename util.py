import matplotlib
import matplotlib.figure
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep

from .SetupConfig import config

from typing import Union, List
from .Cut import AbstractCut, common_cuts, NoCut
from .place_text import place_text

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

def add_cms_legend(ax, isdata: bool):
    if isdata:
        hep.cms.label(ax=ax, data=True, 
                      lumi=config.get('lumi', None),
                      year=config.get('year', None),
                      label=config['cms_label'])
    else:
        hep.cms.label(ax=ax, data=False, 
                      label=config['cms_label'])
        
def savefig(fig, path: str, mkdir : bool=True):
    if mkdir:
        import os
        dirname = os.path.dirname(path)
        if dirname != '':
            os.makedirs(dirname, exist_ok=True)

    print("Saving figure %s" % (path))
    fig.savefig(path+'.png', dpi=300, bbox_inches='tight', format='png')
    fig.savefig(path+'.pdf', bbox_inches='tight', format='pdf')

def ensure_same_length(*args):
    result = []
    for arg in args:
        if isinstance(arg, list):
            result.append(arg)
        else:
            result.append([arg])
    
    maxlen = max([len(x) for x in result])

    for i in range(len(result)):
        if len(result[i]) == 1:
            result[i] = result[i] * maxlen
        elif len(result[i]) != maxlen:
            raise ValueError("All input arguments must have the same length or be of length 1")

    return result

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