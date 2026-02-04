# simonplot  <!-- omit from toc -->

This package provides common matplotlib utilities that I use across several projects.

## Table of contents <!-- omit from toc -->

- [Concepts](#concepts)
- [Configuration](#configuration)
- [Detailed documentation](#detailed-documentation)
  - [Drivers](#drivers)
    - [plot\_histogram()](#plot_histogram)
    - [scatter\_2d()](#scatter_2d)
    - [draw\_matrix()](#draw_matrix)
    - [draw\_radial\_histogram()](#draw_radial_histogram)
  - [Variables](#variables)
    - [Unbinned variables](#unbinned-variables)
    - [Prebinned variables](#prebinned-variables)
  - [Cuts](#cuts)
    - [Unbinned cuts](#unbinned-cuts)
    - [Prebinned cuts](#prebinned-cuts)
  - [Datasets](#datasets)
  - [Binnings](#binnings)
- [Useage example](#useage-example)


## Concepts

This package is built on a few fundamental abstractions:

 - **datasets** abstract data sources and implement access and selection operations, track dataset-level weights (eg for lumi-reweighting), etc. These can be both pre-binned (ie histograms) or unbinned (eg root files, parquet dataset, ...).
 - **binnings** control x-axis binnings and ticks
 - **variables** represent operations on dataset columns
 - **cuts** represent selection operations on variables

I then define a few driver functions that take as inputs datasets, variables, etc and automatically create and format the appropriate plots. The currently-supported drivers are
 - `plot_histogram()`
 - `scatter_2d()`
 - `draw_matrix()`
 - `draw_radial_histogram()`

## Configuration

All of the configuration is handled globally with config json. Defaults are in `config/default.json`, and will be automatically overwritten by values supplied in `simon_mpl_config.json` in your working directory. Documentation of the config can be found in `config/docs.md`. 

## Detailed documentation

### Drivers

#### plot_histogram()

`plot_histogram()` draws one-dimensional histograms. The signature looks like
```python
plot_histogram(
    variable,      # variable or List[variable] to plot
    cut,           # cut or List[cut] to apply before plotting
    weight,        # variable or List[variable] for event weights
    dataset,       # dataset or List[dataset] to use as data source(s)
    binning,       # binning to use for x-axis
    labels,        # List[str] or None. If not None, override labels on automatic legend
    extratext,     # str or None. If not None, extra text to print on the plot
    density,       # bool. Whether to normalize the 1d histograms to have integral 1
    logx,          # bool. Whether to use a logarithmic x-axis
    logy,          # bool. Whether to use a logarithmic y-axis  
    pulls,         # bool. Whether the ratiopad should be normalized by uncertainty
    no_ratiopad,   # if true, force no ratiopad
    output_folder, # destination folder for plot 
    output_prefix  # prefix for plot name on disk
)
```

If any of `variable`, `cut`, `weight`, `dataset` are lists with length > 1, this function will plot multiple histograms with identical x-axis binnings, and a ratiopad. 

`plot_histogram()` has the following automatic behaviors:
 - Detection of the presence of data and MC datasets, including:
    - Calculation of the appropriate cross-section weights
    - Automatically putting data in the numerator for the ratiopad
 - Detection of the presence of dataset stacks, including:
   - If exactly one stack is in the list of datasets, it will be plotted as a stacked histogram
   - Otherwise, the stacks will be plotted as total histograms
 - Automatic placement of text on the plot describing the applied cuts
 - Automatic well-formatted axis labels
 - Automatic naming of .pdf and .png files with all the relevant information
 - Automatic placement of the label and text to minimize overlap with other plot elements
 - Automatic resolution of perverse choices by matplotlib on y-axis limits 
 - Automatic scaling of y-axis limits in the ratiopad to sometimes clip large ratios with large error bars so that features with small error bars are visible 

#### scatter_2d()

Documentation TBD

#### draw_matrix()

Documentation TBD

#### draw_radial_histogram()

Documentation TBD

### Variables

There are two conceptually different kinds of variables, depending on whether the data source is binned (ie a histogram) or unbinned (eg parquet dataset, NANO root file, ...). 

#### Unbinned variables

Unbinned variables satisfy the following protocol:
```python
class UnbinnedVariableProtocol:
        @property
    def prebinned(self) -> bool:
        ...

    @property
    def columns(self) -> List[str]:
        ...

    @property
    def label(self) -> str:
        ...

    def set_collection_name(self, collection_name : str) -> None:
        ...

    def __eq__(self, other) -> bool:
        ...

    def evaluate(self, dataset : "UnbinnedDatasetAccessProtocol", cut : CutProtocol) -> Any:
        ...

    @property
    def centerline(self) -> None | float | Sequence[float]:
        ...
```

By default I have provided the following implementations:
 - `ConstantVariable` - returns a constant
 - `BasicVariable` - just a column from the data source
 - `AkNumVariable` - `awkward.num()` of a column from the data source
 - `RatioVariable` - ratio of two variables
 - `ProductVariable` - product of two variables
 - `DifferenceVariable` - difference of two variables
 - `SumVariable` - sum of two variables
 - `CorrectionlibVariable` - variable computed by an arbitrary function from correctionlib
 - `UFuncVariable` - variable computed by a numpy ufunc
 - `ConcatVariable` - append multiple variables into one long column 
 - `RelativeResolutionVariable` - (reco - gen)/gen
 - `Magnitude3dVariable` - |(x, y, z)|
 - `Magnitude2dVariable` - |(x, y)|
 - `Distance3dVariable` - |vec1 - vec2|
 - `EtaFromXYZVariable` - eta(x, y, z)
 - `PhiFromXYZVariable` - phi(x, y, z)

These can be arbitrarily composed and combined to produce an arbitrary Variable

#### Prebinned variables
Prebinned variables satisfy the following protocol:
```python
class PrebinnedVariableProtocol:
        @property
    def prebinned(self) -> bool:
        ...

    @property
    def columns(self) -> List[str]:
        ...

    @property
    def label(self) -> str:
        ...

    def set_collection_name(self, collection_name : str) -> None:
        ...

    def __eq__(self, other) -> bool:
        ...

    def evaluate(self, dataset : "PrebinnedDatasetAccessProtocol", cut : PrebinnedOperationProtocol) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        ...

    @property
    def hasjacobian(self) -> bool:
        ...

    @property
    def jac_details(self) -> dict[str, Any]:
        ...

    @property
    def normalized_blocks(self) -> List[str]:
        ...
    
    @property
    def normalized_by_err(self) -> bool:
        ...
        
    @property
    def centerline(self) -> None | float | Sequence[float]:
        ...
```

By default I have provided the following implementations:

 - `BasicPrebinnedVariable` - just the values from the histogram
 - `WithJacobian` - normalized by some jacobian
 - `NormalizePerBlock` - normalize by the total integral in blocks (eg for a two-dimensional binning, normalize to have unit integral in each bin of the first axis)
 - `CorrelationFromCovariance` - convert (val, covariance) -> (val/err, correlation)

These can again be arbitrarily composed for arbitrary behavior.

### Cuts

#### Unbinned cuts

Unbinned cuts satisfy the following protocol:
```python
class CutProtocol:
    @property
    def prebinned(self) -> bool:
        ...

    @property
    def columns(self) -> List[str]:
        ...

    @property
    def label(self) -> str:
        ...

    def set_collection_name(self, collection_name : str) -> None:
        ...

    def __eq__(self, other) -> bool:
        ...

    def evaluate(self, dataset : "UnbinnedDatasetAccessProtocol") -> Any:
        ...
```
I have provided the following cuts:
 - `NoCut` - no cut
 - `EqualsCut` - check is a given variable is exactly a given value
 - `TwoSidedCut` - require that a variable is in the range low <= value < high
 - `GreaterThanCut` - require that a variable is >= a value
 - `LessThanCut` - require that a variable is < a value
 - `NotCut` - invert a given cut
 - `AndCuts` - combine multiple cuts with logical AND
 - `OrCuts` - combine multiple cuts with logical OR
 - `ConcatCut` - concatinate multiple cuts into one long column. For use with `ConcatVariable`s 

#### Prebinned cuts

Prebinned cuts apply projection and slicing operations to prebinned histograms. These are also known as `PrebinnedOperation`s, and satisfy the following protocol:
```python
class PrebinnedOperationProtocol:
    @property
    def prebinned(self) -> bool:
        ...

    @property
    def columns(self) -> List[str]:
        ...

    @property
    def label(self) -> str:
        ...

    def set_collection_name(self, collection_name : str) -> None:
        ...

    def __eq__(self, other) -> bool:
        ...

    def evaluate(self, dataset : "UnbinnedDatasetAccessProtocol") -> Any:
        ...
        
    def resulting_binning(self, binning : ArbitraryBinning) -> ArbitraryBinning:
        ...

    def _compute_resulting_binning(self, binning : ArbitraryBinning) -> ArbitraryBinning:
        ...
```
I have provided the following implementations:
 - `NoopOperation` - just take the histogram from the `Variable`
 - `ProjectionOperation` - project down onto the supplied list of axes
 - `SliceOperation` - select a few bins in one or more axes
 - `ProjectAndSliceOperation` - compose projection and slice operations

### Datasets 

Datasets abstract data sources, and have a lot of functionality, including
 - data access
 - projection/slice operations for prebinned operations
 - tracking of lumi/cross-section and weighting
 - tracking of plotting properties (color, label, etc)
 - histogram plotting functionality
 
 I won't reproduce the full interface protocol here. 

 I have provided the following unbinned implementations:

 - `NanoEventsDataset` - read from NANOAOD-formatted root file
 - `ParquetDataset` - read from parquet dataset

And the following prebinned implementations:
  - `ValCovPariDataset` - track prebinned (value, covariance) pairs

And a stack implementation:
 - `StackDataset` - form a dataset as a composition of multiple datasets, which can be either prebinned or unbinned, but not both

### Binnings

Binnings represent how to determine the x-axis binning from the dataset. There are a few options:

 - `AutoBinning` - automatically chose the binning with a heuristic.
 - `AutoIntCategoryBinning` - automatically create the binning for integer category variables (eg pdgid). This includes an optional `label_lookup` field which can map integer category labels (eg 211) to string labels for the plot (eg "pi+").
 - `DefaultBinning` - read the binning from a config file, looked up by the name of the variable being plotted.
 - `BasicBinning` - build a binning from `nbins`, `low`, `high`
 - `ExplicitBinning` - build a binning from the provided edges
 - `PrebinnedBinning` - basically just a placeholder object for prebinned data, where the binning is pre-defined. 

## Useage example

```python
from simonplot.variable import BasicVariable, RatioVariable
from simonplot.cut import TwoSidedCut, GreaterThanCut, AndCuts
from simonplot.plottables import NanoEventsDataset, StackDataset
from simonplot.binning import BasicBinning, AutoIntCategoryBinning
from simonplot.drivers import plot_histogram

# Define variables
pt = BasicVariable('jet_pt', label=r'$p_T$ [GeV]')
eta = BasicVariable('jet_eta', label=r'$\eta$')
phi = BasicVariable('jet_phi', label=r'$\phi$')
pdgid = BasicVariable('pdgid', label='Particle ID')

# Define a composite variable
pt_over_mass = RatioVariable(
    BasicVariable('jet_pt'), 
    BasicVariable('jet_mass'),
    label=r'$p_T / m$'
)

# Define cuts
eta_cut = TwoSidedCut(eta, -2.5, 2.5, label=r'$|\eta| < 2.5$')
pt_cut = GreaterThanCut(pt, 30.0, label=r'$p_T > 30$ GeV')
combined_cut = AndCuts([eta_cut, pt_cut])

# Create datasets
data = NanoEventsDataset(
    'data.root',
    label='Data',
    color='black',
    luminosity=138.0      # fb^-1

)

mc_signal = NanoEventsDataset(
    'signal_mc.root',
    label='Signal MC',
    color='red',
    cross_section=100.0,  # pb
)

mc_background = NanoEventsDataset(
    'background_mc.root',
    label='Background MC',
    color='blue',
    cross_section=500.0,
)

# Create a stack of MC datasets
mc_stack = StackDataset([mc_signal, mc_background], label='MC')

# Define binning
pt_binning = BasicBinning(nbins=50, low=0, high=500)
pdgid_binning = AutoIntCategoryBinning(
    label_lookup={
        211: r'$\pi^+$',
        -211: r'$\pi^-$',
        321: r'$K^+$',
        -321: r'$K^-$'
    }
)

# Plot histogram comparing data and MC
plot_histogram(
    variable=pt,
    cut=combined_cut,
    weight=None,
    dataset=[data, mc_stack],
    binning=pt_binning,
    extratext='Preliminary',
    logy=True,
    output_folder='plots',
    output_prefix='jet_pt_comparison'
)

# Plot particle ID distribution with categorical binning
plot_histogram(
    variable=pdgid,
    cut=pt_cut,
    weight=None,
    dataset=mc_signal,
    binning=pdgid_binning,
    output_folder='plots',
    output_prefix='particle_ids'
)
```
