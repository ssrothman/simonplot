import matplotlib.pyplot as plt
from .SetupConfig import config
import mplhep as hep
hep.style.use(hep.style.CMS)

def setup_canvas():
    fig = plt.figure(figsize=config['figsize'])
    ax = fig.add_subplot(1,1,1)

    return fig, ax

def add_cms_legend(ax, isdata: bool):
    if isdata:
        hep.cms.label(ax=ax, data=True, 
                      lumi=config.get('lumi', None),
                      year=config.get('year', None),
                      label=config['cms_label'])
    else:
        hep.cms.label(ax=ax, data=False, 
                      label=config['cms_label'])
        
