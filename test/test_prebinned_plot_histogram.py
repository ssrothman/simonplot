from data_factory import synthetic_valcovdataset
from simon_mpl_util.plotting.plottables import DatasetStack
from simon_mpl_util.plotting import plot_histogram
from simon_mpl_util.plotting.variable import BasicPrebinnedVariable, ConstantVariable, WithJacobian, NormalizePerBlock
from simon_mpl_util.plotting.cut import NoopOperation, ProjectionOperation, SliceOperation, ProjectAndSliceOperation
from simon_mpl_util.plotting.binning import PrebinnedBinning

dset_MC1, _, _ = synthetic_valcovdataset(100000, "MCx1")
dset_MC2, _, _ = synthetic_valcovdataset(200000, "MCx2")
dset_MC1.set_xsec(1000.0)
dset_MC2.set_xsec(2000.0)
dset_MC1.set_color('blue')
dset_MC2.set_color('orange')

dset_MCstack = DatasetStack("MC Stack", None, "MC-Stack", [dset_MC1, dset_MC2])
dset_MCstack.set_color('green')

dset_data, _, _ = synthetic_valcovdataset(300000, "data")
dset_data.set_lumi(0.1)
dset_data.set_color('black')

var1 = BasicPrebinnedVariable()
binning = PrebinnedBinning()

cut1 = NoopOperation()
cut2 = ProjectionOperation(['pt'])
cut3 = ProjectionOperation(['r'])
cut4 = ProjectionOperation(['pt', 'phi'])
cut5 = SliceOperation({'pt' : (dset_MC1.binning.edges['pt'][2], dset_MC1.binning.edges['pt'][5])})
cut6 = SliceOperation({'r' : (dset_MC1.binning.edges['r'][0], dset_MC1.binning.edges['r'][1]),
                       'phi' : (dset_MC1.binning.edges['phi'][3], dset_MC1.binning.edges['phi'][7])})
cut7 = ProjectAndSliceOperation(
    axes = ['r'],
    edges = {'pt' : (dset_MC1.binning.edges['pt'][1], dset_MC1.binning.edges['pt'][2])}
)
cut8 = ProjectionOperation(['r', 'phi'])

weight = ConstantVariable(1.0)

for dsetset in [dset_MC1,  [dset_MC1, dset_MC2],  [dset_MC1, dset_MC2, dset_MCstack, dset_data],  [dset_MCstack, dset_data],  [dset_MCstack, dset_data]]: 
#for dataset in []:
    plot_histogram(
        var1,
        cut1,
        weight,
        dsetset,
        binning,
        output_folder='unittest/prebinned_plot_histogram/hist',
        logy = True,
    )

for cut in [cut2, cut3, cut4, cut5, cut6, cut7, cut8]:
#for cut in []:
     plot_histogram(
         var1,
         cut,
         weight,
         [dset_MCstack, dset_data],
         binning,
         output_folder='unittest/prebinned_plot_histogram/hist',
         logy = True,
     )


var2 = WithJacobian(var1, radial_coords=['r'], clip_negativeinf={'pt' : 0.0}, clip_positiveinf={'pt' : 10000.0})
var3 = NormalizePerBlock(var1, axes=['pt'])
var4 = NormalizePerBlock(var1, axes=['pt', 'r'])
var5 = WithJacobian(var3, radial_coords=['r'], clip_negativeinf={'pt' : 0.0}, clip_positiveinf={'pt' : 10000.0})

for var in [var1, var2, var3, var4, var5]:
#for var in [var4]:
     plot_histogram(
        var, 
        cut1,
        weight,
        [dset_MC1, dset_MC2],
        binning,
        output_folder='unittest/prebinned_plot_histogram/hist',
        logy = True,
     )