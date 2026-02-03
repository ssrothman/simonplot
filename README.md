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
- [1. Quickstart](#1-quickstart)
  - [1.3 Making plots](#13-making-plots)
  - [1.4 Controlling the plots](#14-controlling-the-plots)
    - [1.4.1 Variable names](#141-variable-names)
    - [1.4.2 Nontrivial variables](#142-nontrivial-variables)
- [2. Variables](#2-variables)
  - [2.1 Spec](#21-spec)
  - [2.2 Provided implementations](#22-provided-implementations)
    - [2.2.1 Basic Variables](#221-basic-variables)
    - [2.2.2 ak.num() variables](#222-aknum-variables)
    - [2.2.3 Arithmatic on Variables](#223-arithmatic-on-variables)
    - [2.2.4 Correctionlib functions](#224-correctionlib-functions)
    - [2.2.5 UFuncs](#225-ufuncs)
    - [2.2.6 Rates](#226-rates)
    - [2.2.7 More complex Variables](#227-more-complex-variables)
- [3. Datasets](#3-datasets)
  - [3.1 Spec](#31-spec)
  - [3.2 Provided implementations](#32-provided-implementations)
    - [3.2.1 NanoEventsDataset](#321-nanoeventsdataset)
    - [3.2.2 ParquetDataset](#322-parquetdataset)
- [4. Cuts](#4-cuts)
- [5. Binning](#5-binning)
- [6. simon\_mpl\_config.json](#6-simon_mpl_configjson)


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

## 1. Quickstart

### 1.3 Making plots

The main entry points into this package are the `plot_histogram()` and `scatter_2d()` functions. Both expect `Variable`, `Cut`, and `Dataset` parameters to tell them what to plot. The `plot_histogram` function also requires a `Binning` object to tell it what binning to use. A basic example might look like

```python
import simon_mpl_util as smu

dataset =smu.NanoEventsDataset("NANO.root:Events")
var = smu.Variable('genPart.pt')
cut = smu.NoCut()
binning = smu.AutoBinning()

smu.plot_histogram(
    var, cut, dataset, binning
)

x = smu.Variable('genPart.eta')
y = smu.Variable('genPart.phi')
cut = smu.GreaterThanCut('genPart.pt', 20)

smu.scatter_2d(
    x, y,
    cut, dataset
)
```

This will plot a histogram of the genParticle pT, and then a 2d scatter plot of the genParticle eta and phi coordinates. Use of the `AutoBinning()` will tell the code to try to inteligently come up with a reasonable binning to use. Additionally, the code will automatically place a text box on the scatter plot describing the applied selection. 

### 1.4 Controlling the plots

#### 1.4.1 Variable names

By default the variable names on the axis labels and selection text boxes will be exactly the names of the variables in the nano. In order to avoid having to copy the same axis labels throughout the code, replacement rules for these names are controlled in the config file `simon_mpl_config.json` through the `axis_labels` dictionary. This might look like
```json
#simon_mpl_config.json
{
    "axis_labels": {
        "*.pt": "$p_{T}$ [GeV]",
        "*.eta": "$\\eta$",
        "*.phi": "$\\phi$",
        "*.x" : "x [cm]",
        "*.y" : "y [cm]",
        "*.z" : "z [cm]",
        ...
    },
    ...
}
```

The keys in this dictionary are variable names (these can be fields in the NANO for simple variables, or more complicated compound expressions), and the corresponding values are the string replacements to be used in the plot. In order to support similar fields across multiple NANO collections (eg `genPart.pt` and `Jet.pt`) the `*` wildcard is supported for the key matching. All other special characters (including, importantly `.`s are treated literally). Try adding these fields to your config json and rerunning the previous code - hopefully the plots are a lot more appealing now. 

If you don't want to add a row to your json, you can also override the default label behavior of a given `Variable` object with the `override_label(label)` method. This can be reverted with the `clear_override_label()` method. 

#### 1.4.2 Nontrivial variables

In addition to the simple `Variable` class which directly refers to columns in the NANO file, many compound operations are also supported. These include:

 - The sum of two `Variables` (`SumVariable`)
 - The difference of two `Variables` (`Difference Variable`)
 - The product of two `Variables` (`ProductVariable`)
 - The ratio of two `Variables` (`RatioVariable`)
 - Functions of `Variables` as implemented by numpy UFuncs (`UFunc Variable`)
 - Functions of `Variables` as implemented by correctionlib (`CorrectionlibVariable`)
 - Rates for binary yes/no variables (`RateVariable`)

These classes can be composed arbitrarily, for example to create the provided `Magnitude3dVariable` for the magnitude of 3-dimensional vectors. 

Further, the `Cut` classes accept arbitrary `Variable`s as input. So one could, for example, plot a histogram with a selection on the 3-dimensional distance from the origin to the gen vertex by using a `Magnitude3dVariable`. 

This is all discussed in greater detail in the `Variables` section of this README. 

## 2. Variables

### 2.1 Spec

All `Variables` inherit from and obey the interface of the `AbstractVariable` class:
```python
class AbstractVariable:
    @property
    def columns(self):
        raise NotImplementedError()

    def evaluate(self, table):
        raise NotImplementedError()

    @property
    def key(self):
        raise NotImplementedError()
    
    def override_label(self, label):
        self._label = label

    def clear_override_label(self):
        if hasattr(self, '_label'):
            del self._label

    @property
    def label(self):
        if hasattr(self, '_label') and self._label is not None:
            return self._label
        else:
            return lookup_axis_label(self.key)

    def __eq__(self, other):
        raise NotImplementedError()
```

Any `Variable` class must implement the abstract methods of this interface:

 - The `columns` property must return a list of columns which must be available for evaluation of the variable
 - The `key` property is a unique name identifying the variable (for example in the `axis_labels` dictionary)
 - The `evalute()` method evaluates the variable to an numpy or awkward array

### 2.2 Provided implementations

#### 2.2.1 Basic Variables

A `Variable` object just points to a column in the dataset. The `evaluate()` method is farmed off to the dataset implementation to allow for different access syntax depending on the underlying data format. 

#### 2.2.2 ak.num() variables

When dealing with NANO datasets, it is useful to be able to access how many of a given object there are per event. This is implemented with the `AkNumVariable` class. Note that as currently implemented this calls `ak.num()` BEFORE any selection cuts. This will probably need to be rethought. 

#### 2.2.3 Arithmatic on Variables

The four basic artithmetic operations are supported between any subclasses of `AbstractVariable` with the `SumVariable`, `DifferenceVariable`, `ProductVariable`, and `RatioVariable` classes. These can be composed arbitrarily to create complex mathematical operations. 

#### 2.2.4 Correctionlib functions

Functions implemented by correctionlib are supported on `AbstractVariable`s through the `CorrectionlibVariable` class. The constructor for this class looks like `CorrectionlibVariable(var_l, path, key)`, where:

 - `var_l` is the (ordered) list of `AbstractVariable`s to pass to the correctionlib function
 - `path` is the path in the filesystem to the correctionlib correctionset file
 - `key` is the name of the function in the correctionset to use

#### 2.2.5 UFuncs 

Numpy provides so-called "universal functions" on variables such as `np.sqrt`, `np.exp`, and `np.log`. These are supported by the `UFuncVariable` class. The constructor for this class looks like `UFuncVariable(var, ufunc)` where
  
  - `var` is the `AbstractVariable` input to the ufunc
  - `ufunc` is the python object for the ufunc

For example, a square root can be implemented as 

```python
sqrtX = smu.UFuncVariable(
    smu.Variable('genVtx.x'),
    np.sqrt #note the lack of ()!
)
```

#### 2.2.6 Rates

It can be a bit annoying to go from a binary yes/no variable to a rate as a function of some other variable. This is supported by the `RateVariable` class. The constructor for this class looks like `RateVariable(binaryfield, wrt)`, where:

 - `binaryfield` is the `AbstractVariable` for the 1/0 binary variable that you want the rate of
 - `wrt` is the `AbstractVariable` that you want the rate as a function of 

Note that both of these can be arbitrary `AbstractVariable`s.

#### 2.2.7 More complex Variables

In principle all of the previous ingredients are enough to create nearly any function with enough effort. I have provided pre-writted versions of some more commonly-needed composite variables. These are:

 - `RelativeResolutionVariable` implements `(reco - gen)/gen`
 - `Magnitude3dVariable` implements `|(x, y, z)|`
 - `Distance3dVariable` implements `|(x1, y1, z1) - (x2, y2, z2)|`

## 3. Datasets

The unerlying data storage format is abstracted a `dataset` object, allowing the code to be completely general. 

### 3.1 Spec

All datasets must inherit the `AbstractDataset` class, which looks like
```python
class AbstractDataset:
    def ensure_columns(self, columns):
        raise NotImplementedError()

    def get_column(self, column_name):
        raise NotImplementedError()

    def get_aknum_column(self, column_name):
        raise NotImplementedError()
    
    @property
    def num_rows(self):
        raise NotImplementedError()
```
These methods are:
 - `ensure_columns(columns)` must ensure that the columns listed in the input are avilable for evaluation. Depending on the implementation of the data storage this may be a no-op
 - `get_column(column_name)` must return the column pointed to be `column_name` in a numpy or awkward array
 - `get_aknum_column(column_name)` must return the number of objects per event named by `column_name`. This only really makes sense for NANO backends, and can be left unimplemented if you don't intend to ever construct `AkNumVariable`s 
 - `num_rows` must return the number of rows in the dataset

### 3.2 Provided implementations

#### 3.2.1 NanoEventsDataset 

`NanoEventsDataset`s read CMS NANOAOD files. They provide an implementation of get_aknum_column() as well as the standard methods. The constructor looks like `NanoEventsDataset(fname, **options)`, where 
 - `fname` is the filename to read, with the treepath (usually `Events`) appended after a `:`. For example, it might be `NANO_selected.root:Events`. 
 - `**options` are arbitrary kwargs options that are passed along to the coffea `NanoEventsFactory` `from_root` method. 

#### 3.2.2 ParquetDataset

`ParquetDataset`s read datasets stored as a folder of 

## 4. Cuts 

Cuts abstract selections. 

## 5. Binning

## 6. simon_mpl_config.json
