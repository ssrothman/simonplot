import matplotlib.pyplot as plt
import hist
import numpy as np

from .AribtraryBinning import ArbitraryBinning

def _call_errorbar(ax, x, y, xerr, yerr, **kwargs):
    return ax.errorbar(
        x, y, xerr = xerr, yerr = yerr,
        fmt = 'o', markersize=4, capsize=1, 
        **kwargs
    )

def _call_stairs(ax, edges, plotvals, fillbetween, **kwargs):
    return ax.stairs(
        plotvals, edges, 
        baseline = fillbetween, fill=True, 
        **kwargs
    )

def simon_histplot_rate(H, ax=None, **kwargs):
    if len(H.axes) != 2:
        raise ValueError("histplot_rate only supports 2D histograms")

    if ax is None:
        ax = plt.gca()

    vals = H.values().copy()
    errs = np.sqrt(H.variances()).copy()

    passvals = vals[1, :]
    passerrs = errs[1, :]

    failvals = vals[0, :]
    failerrs = errs[0, :]

    total = passvals + failvals

    rate = passvals / total
    rateerr = np.sqrt(
        np.square((1-rate)*passerrs/total) +
        np.square(rate*failerrs/total)
    )

    edges = H.axes[1].edges
    centers = H.axes[1].centers
    widths = H.axes[1].widths

    if type(H.axes[1]) is hist.axis.Integer:
        centers -= 0.5
        edges -= 0.5

    return _call_errorbar(ax, centers, rate, widths/2, rateerr, **kwargs)

def _simon_histplot(vals, errs, edges, centers, widths,
                    ax=None, density=False, fillbetween = None, **kwargs):
    if density:
        N = np.sum(vals)
        vals /= N
        errs /= N

    plotvals = vals/widths
    ploterrs = errs/widths

    if fillbetween is not None:
        artist = _call_stairs(ax, edges, plotvals+fillbetween, fillbetween, **kwargs)
        return artist, plotvals+fillbetween
    else:
        artist = _call_errorbar(ax, centers, plotvals, widths/2, ploterrs, **kwargs)
        return artist, plotvals

def simon_histplot_arbitrary(vals : np.ndarray, cov : np.ndarray,
                             binning : ArbitraryBinning, 
                             ax=None, density=False, fillbetween = None, **kwargs):
    
    errs = np.sqrt(np.diag(cov))

    if binning.Nax == 1:
        axname = binning.axis_names[0]
        edges = binning.edges[axname]

    else:
        edges = np.arange(len(vals)+1)-0.5

    centers = (edges[:-1] + edges[1:]) / 2
    widths = edges[1:] - edges[:-1]

    return _simon_histplot(vals, errs, edges, centers, widths,
                           ax=ax, density=density, fillbetween=fillbetween, **kwargs)

def simon_histplot(H, ax=None, density=False, fillbetween = None, **kwargs):
    if len(H.axes) != 1:
        raise ValueError("histplot only supports 1D histograms")

    if ax is None:
        ax = plt.gca()

    vals = H.values().copy()
    errs = np.sqrt(H.variances()).copy()

    edges = H.axes[0].edges
    centers = H.axes[0].centers
    widths = H.axes[0].widths

    if type(H.axes[0]) is hist.axis.Integer:
        centers -= 0.5
        edges -= 0.5

    return _simon_histplot(vals, errs, edges, centers, widths,
                           ax=ax, density=density, fillbetween=fillbetween, **kwargs)

def _simon_histplot_ratio(vals_num, errs_num,
                          vals_denom, errs_denom,
                          edges, centers, widths,
                          ax=None, 
                          density=False, pulls=False, **kwargs):
    if density:
        Nnum = np.sum(vals_num)
        Ndenom = np.sum(vals_denom)

        errs_num /= Nnum
        vals_num /= Nnum

        errs_denom /= Ndenom
        vals_denom /= Ndenom

    vals_num /= widths
    errs_num /= widths

    vals_denom /= widths
    errs_denom /= widths

    with np.errstate(divide='ignore', invalid='ignore'): #ignore warnings from 0/0 operations. These return NaN, which are handled correctly downstream
        ratio = vals_num / vals_denom

        ratio_err = np.sqrt(
                np.square(errs_num/vals_num) + np.square(errs_denom/vals_denom)
        ) * ratio

    if pulls:
        ratio = ratio-1
        ratio = ratio/ratio_err
        ratio_err = np.ones_like(ratio)
    
    return _call_errorbar(ax, centers, ratio, widths/2, ratio_err, **kwargs), ratio, ratio_err

def simon_histplot_ratio_arbitrary(num, denom,
                                   binning : ArbitraryBinning,
                                   ax=None,
                                   density=False, pulls=False, **kwargs):
    vals_num, cov_num = num
    vals_denom, cov_denom = denom
    
    errs_num = np.sqrt(np.diag(cov_num))
    errs_denom = np.sqrt(np.diag(cov_denom))

    if binning.Nax == 1:
        axname = binning.axis_names[0]
        edges = binning.edges[axname]
    else:
        edges = np.arange(len(vals_num)+1)-0.5

    centers = (edges[:-1] + edges[1:]) / 2
    widths = edges[1:] - edges[:-1]

    return _simon_histplot_ratio(vals_num, errs_num,
                                 vals_denom, errs_denom,
                                 edges, centers, widths,
                                 ax=ax, density=density, pulls=pulls, **kwargs)

def simon_histplot_ratio(Hnum, Hdenom, ax=None, 
                         density=False, pulls=False, **kwargs):
    
    if len(Hnum.axes) != 1 or len(Hdenom.axes) != 1:
        raise ValueError("histplot only supports 1D histograms")

    if Hnum.axes[0] != Hdenom.axes[0]:
        raise ValueError("histograms must have the same axes")

    if ax is None:
        ax = plt.gca()

    vals_num = Hnum.values().copy()
    errs_num = np.sqrt(Hnum.variances()).copy()

    vals_denom = Hdenom.values().copy()
    errs_denom = np.sqrt(Hdenom.variances()).copy()

    edges = Hnum.axes[0].edges 
    centers = Hnum.axes[0].centers
    widths = Hnum.axes[0].widths

    if type(Hnum.axes[0]) is hist.axis.Integer:
        centers -= 0.5

    return _simon_histplot_ratio(vals_num, errs_num,
                                 vals_denom, errs_denom,
                                 edges, centers, widths,
                                 ax=ax, density=density, pulls=pulls, **kwargs)