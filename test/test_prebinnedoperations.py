from data_factory import synthetic_valcovdataset, synthetic_covdataset
from printing import print_details, assertions_valcov, assertions_covmat
import numpy as np


from simonplot.variable import BasicPrebinnedVariable
from simonplot.cut import NoopOperation, ProjectionOperation, SliceOperation, ProjectAndSliceOperation

def check_operations(dset, target_vals, target_cov, assertfun):
    var1 = BasicPrebinnedVariable()

    cut1 = NoopOperation()
    cut2 = ProjectionOperation(['pt'])
    cut3 = ProjectionOperation(['r'])
    cut4 = ProjectionOperation(['pt', 'phi'])
    cut5 = SliceOperation({'pt' : (dset.binning.edges['pt'][2], dset.binning.edges['pt'][5])}, clipemptyflow=[])
    cut6 = SliceOperation({'r' : (dset.binning.edges['r'][0], dset.binning.edges['r'][1]),
                        'phi' : (dset.binning.edges['phi'][3], dset.binning.edges['phi'][7])}, clipemptyflow=[])
    cut7 = ProjectAndSliceOperation(
        axes = ['r'],
        edges = {'pt' : (dset.binning.edges['pt'][1], dset.binning.edges['pt'][2])},    
        clipemptyflow = []
    )

    result1 = var1.evaluate(dset, cut1)
    assertfun(result1, dset.data)
    print("\tNoop operation passed checks!")

    result2 = var1.evaluate(dset, cut2)
    targetlen = target_cov.shape[1] * target_cov.shape[2]
    target_projpt = np.sum(target_cov, axis=(0, 3)).ravel().reshape((targetlen, targetlen))
    if target_vals is not None:
        target_projpt_vals = np.sum(target_vals, axis=0).ravel()
        target_projpt = (target_projpt_vals, target_projpt)
    assertfun(result2, target_projpt)
    print("\tProjection operation (pt) passed checks!")

    result3 = var1.evaluate(dset, cut3)
    targetlen = target_cov.shape[0] * target_cov.shape[2]
    target_projr = np.sum(target_cov, axis=(1, 4)).ravel().reshape((targetlen, targetlen))
    if target_vals is not None:
        target_projr_vals = np.sum(target_vals, axis=1).ravel()
        target_projr = (target_projr_vals, target_projr)
    assertfun(result3, target_projr)
    print("\tProjection operation (r) passed checks!")

    result14 = var1.evaluate(dset, cut4)
    targetlen = target_cov.shape[1]
    target_projptr = np.sum(target_cov, axis=(0, 2, 3, 5)).ravel().reshape((targetlen, targetlen))
    if target_vals is not None:
        target_projptr_vals = np.sum(target_vals, axis=(0, 2), keepdims=True).ravel()
        target_projptr = (target_projptr_vals, target_projptr)
    assertfun(result14, target_projptr)
    print("\tProjection operation (pt, phi) passed checks!")

    result15 = var1.evaluate(dset, cut5)
    targetlen = (5 - 2) * target_cov.shape[1] * target_cov.shape[2]
    target_slicept = target_cov[2:5, :, :, 2:5, :, :].reshape((targetlen, targetlen))
    if target_vals is not None:
        target_slicept_vals = target_vals[2:5, :, :].ravel()
        target_slicept = (target_slicept_vals, target_slicept)
    assertfun(result15, target_slicept)
    print("\tSlice operation (pt) passed checks!")

    result16 = var1.evaluate(dset, cut6)
    targetlen = target_cov.shape[0] * (7 - 3) * (1-0)
    target_slicerphi = target_cov[:, 0:1, 3:7, :, 0:1, 3:7].reshape((targetlen, targetlen))
    if target_vals is not None:
        target_slicerphi_vals = target_vals[:, 0:1, 3:7].ravel()
        target_slicerphi = (target_slicerphi_vals, target_slicerphi)
    assertfun(result16, target_slicerphi)
    print("\tSlice operation (r, phi) passed checks!")

    result17 = var1.evaluate(dset, cut7)
    targetlen = (2 -1) * target_cov.shape[2]
    target_projandslice = target_cov[1:2, :, :, 1:2, :, :].sum(axis=(1, 4)).ravel().reshape((targetlen, targetlen))
    if target_vals is not None:
        target_projandslice_vals = target_vals[1:2, :, :].sum(axis=1).ravel()
        target_projandslice = (target_projandslice_vals, target_projandslice)
    assertfun(result17, target_projandslice)
    print("\tProject and Slice operation passed checks!")

    print()
    print("\tAll tests passed!")

print("Building synthetic data...")
dset1, H1_vals, H1_cov = synthetic_valcovdataset(100000, "dset1")
dset2, H2_cov = synthetic_covdataset(100000, "dset2")
print("\tDone.")

print("With valcov dataset...")
check_operations(dset1, H1_vals, H1_cov, assertions_valcov)
print("With covmat dataset...")
check_operations(dset2, None, H2_cov, assertions_covmat)
print("All tests passed!")