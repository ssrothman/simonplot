# simonplot  <!-- omit from toc -->

This package provides common matplotlib utilities that I use across several projects. The core ingredients are:

 - Basic utility functions (defined in util.py)
 - Advanced plotting methods for histograms and 2d scatter plots
 - Control over plot style, labels, etc through a simple API and a json config file
  

## Table of contents <!-- omit from toc -->

- [1. Quickstart](#1-quickstart)
  - [1.1 Setup](#11-setup)
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


## 1. Quickstart

### 1.1 Setup 

Use of this package requires a file named `simon_mpl_config.json` in the working area. Upon import this package will read the config file to setup axis labels, plot styles, etc. The structure of this config file will be described in more detail later, but the only two mandatory fields are `figsize` and `cms_label`. A valid basic config might look like:

```json
{
    "figsize" : [12, 10],
    "cms_label" : "Work in Progress"
}
```

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
