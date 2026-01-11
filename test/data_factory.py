from typing import Tuple
import hist
import numpy as np

from simonplot.plottables import ValCovPairDataset, CovmatDataset
from simonpy.AbitraryBinning import ArbitraryBinning

def synthetic_data(Nevt, key) -> Tuple[np.ndarray, np.ndarray, ArbitraryBinning]:
    H = hist.Hist(
        hist.axis.Regular(7, 30, 500, name='pt', transform=hist.axis.transform.log),
        hist.axis.Regular(10, 0, 1, name='r', underflow=False, overflow=False),
        hist.axis.Regular(10, 0, np.pi/2, name='phi', underflow=False, overflow=False),
        storage=hist.storage.Weight(),
    )

    #draw pT from a power law distribution
    pt = np.random.pareto(a=3.0, size=Nevt) * 50
    r = np.square(np.random.uniform(0, 1, size=Nevt))
    phi = np.random.uniform(0, np.pi/2, size=Nevt)
    w = np.ones(Nevt)

    H.fill(pt=pt, r=r, phi=phi, weight=w)

    vals = H.values(flow=True).ravel()
    cov = np.diag(H.variances(flow=True).ravel()) # pyright: ignore[reportOptionalMemberAccess]
    
    correlations = np.random.uniform(-0.3, 0.8, size=cov.shape)
    err = np.sqrt(np.diag(cov))
    cov_offdiag = np.outer(err, err) * correlations
    np.fill_diagonal(cov_offdiag, 0.0)
    cov += cov_offdiag

    binning = ArbitraryBinning()
    binning.setup_from_histogram(H)

    unflat_vals = H.values(flow=True)
    unflat_cov = cov.reshape(unflat_vals.shape + unflat_vals.shape)

    return unflat_vals, unflat_cov, binning

def synthetic_valcovdataset(Nevt, key) -> Tuple[ValCovPairDataset, np.ndarray, np.ndarray]:
    unflat_vals, unflat_cov, binning = synthetic_data(Nevt, key)
    vals = unflat_vals.ravel()
    cov = unflat_cov.reshape((vals.size, vals.size))

    dset = ValCovPairDataset(key, None, key, (vals, cov), binning)

    return dset, unflat_vals, unflat_cov

def synthetic_covdataset(Nevt, key) -> Tuple[CovmatDataset, np.ndarray]:
    unflat_vals, unflat_cov, binning = synthetic_data(Nevt, key)
    vals = unflat_vals.ravel()
    cov = unflat_cov.reshape((vals.size, vals.size))

    dset = CovmatDataset(key, None, key, cov, binning)

    return dset, unflat_cov