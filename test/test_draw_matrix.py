from data_factory import synthetic_covdataset, synthetic_valcovdataset
from plotting.variable.PrebinnedVariable import CorrelationFromCovariance
from simon_mpl_util.plotting import draw_matrix

from simon_mpl_util.plotting.variable import BasicPrebinnedVariable, ConstantVariable, WithJacobian, NormalizePerBlock
from simon_mpl_util.plotting.cut import NoopOperation, ProjectionOperation, SliceOperation, ProjectAndSliceOperation
from simon_mpl_util.plotting.binning import PrebinnedBinning

dset, _, _= synthetic_valcovdataset(100000, "MC")

var1 = BasicPrebinnedVariable()
binning = PrebinnedBinning()

cut1 = NoopOperation()
cut2 = ProjectionOperation(['pt'])
cut3 = ProjectionOperation(['r'])
cut4 = ProjectionOperation(['pt', 'phi'])
cut5 = ProjectionOperation(['r', 'phi'])
cut6 = SliceOperation({'pt' : (dset.binning.edges['pt'][2], dset.binning.edges['pt'][5])})
cut7 = SliceOperation({'r' : (dset.binning.edges['r'][0], dset.binning.edges['r'][1]),
                       'phi' : (dset.binning.edges['phi'][3], dset.binning.edges['phi'][7])})
cut8 = ProjectAndSliceOperation(
    axes = ['r'],
    edges = {'pt' : (dset.binning.edges['pt'][1], dset.binning.edges['pt'][2])}
)
weight = ConstantVariable(1.0)

#for cut in [cut1, cut2, cut3, cut4, cut5, cut6, cut7, cut8]:
for cut in []:
    draw_matrix(
        var1,
        cut,
        dset,
        binning,
        output_folder = 'unittest/draw_matrix/matrix'
    )

var2 = WithJacobian(var1, radial_coords=['r'], clip_negativeinf={'pt' : 0.0}, clip_positiveinf={'pt' : 10000.0})
var3 = NormalizePerBlock(var1, axes=['pt'])
var4 = NormalizePerBlock(var1, axes=['pt', 'r'])
var5 = WithJacobian(var3, radial_coords=['r'], clip_negativeinf={'pt' : 0.0}, clip_positiveinf={'pt' : 10000.0})
var6 = CorrelationFromCovariance(var1)

#for var in [var2, var3, var4, var5, var6]:
for var in [var4]:
    draw_matrix(
        var,
        cut1,
        dset,
        binning,
        output_folder = 'unittest/draw_matrix/matrix' 
    )