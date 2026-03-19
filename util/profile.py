import numpy as np
import hist
import awkward as ak
from typing import Any, Literal, assert_never
from scipy.stats import binned_statistic

_MODE_OPTIONS = Literal['mean', 'std', 'median', 'sum', 'min', 'max', 'percentile', 'percentile-range']

class ProfileStruct:
    def __init__(self, 
                 xvar : np.ndarray | ak.Array, 
                 yvar : np.ndarray | ak.Array,
                 mode : _MODE_OPTIONS,
                 mode_params : Any = None):
        
        self._xvar = xvar
        self._yvar = yvar

        self._mode = mode
        self._mode_params = mode_params

    @property
    def xvar(self):
        return self._xvar

    @property
    def yvar(self):
        return self._yvar    
    
    @property
    def mode(self):
        return self._mode
    
    @property
    def mode_params(self):
        return self._mode_params

class ProfileHistStruct:

    def __init__(self, 
                 data : ProfileStruct, 
                 axes : Any):
          
        self._data = data
        self._axes = axes
        self._bins = axes[0].edges
        self._mode = data.mode
        self._mode_params = data.mode_params

        if self._mode == 'percentile':
            if not type(self._mode_params) in [int, float]:
                raise ValueError("For 'percentile' mode, mode_params must be a single number representing the desired percentile (0-100)!")
            self._statistic, self._t1, self._t2 = binned_statistic(
                data.xvar,
                data.yvar,
                statistic=lambda y: np.percentile(y, self._mode_params), # pyright: ignore[reportArgumentType]
                bins=self._bins,
            )

        elif self._mode == 'percentile-range':
            if not type(self._mode_params) in [tuple, list] or len(self._mode_params) != 2:
                raise ValueError("For 'percentile-range' mode, mode_params must be a tuple/list of two numbers representing the lower and upper percentiles (0-100)!")
            
            lower, upper = self._mode_params
            if not (0 <= lower < upper <= 100):
                raise ValueError("For 'percentile-range' mode, mode_params must contain two numbers where 0 <= lower < upper <= 100!")
            
            def percentile_range_func(y):
                lower_val = np.percentile(y, lower)
                upper_val = np.percentile(y, upper)
                return upper_val - lower_val
            
            self._statistic, self._t1, self._t2 = binned_statistic(
                data.xvar,
                data.yvar,
                statistic=percentile_range_func, # pyright: ignore[reportArgumentType]
                bins=self._bins,
            )
        else: 
            if self._mode not in ['mean', 'std', 'median', 'sum', 'min', 'max']:
                raise ValueError(f"Unsupported mode {self._mode} for ProfileHistStruct! Supported modes are: 'mean', 'std', 'median', 'sum', 'min', 'max', 'percentile', and 'percentile-range'.")
            
            if self._mode_params is not None:
                raise ValueError(f"Mode {self._mode} does not support mode_params, but got {self._mode_params}!")

            self._statistic, self._t1, self._t2 = binned_statistic(
                data.xvar,
                data.yvar,
                statistic=self._mode,
                bins=self._bins,
            )
            
    @property
    def data(self):
        return self._data
    
    @property
    def axes(self):
        return self._axes
        
    def __add__(self, other):
        if not isinstance(other, ProfileHistStruct):
            raise RuntimeError("ProfileHistStruct.__add__: Can only add another ProfileHistStruct, but got %s!"%type(other))
        
        if self._mode != other._mode or self._mode_params != other._mode_params:
            raise RuntimeError("ProfileHistStruct.__add__: Cannot add ProfileHistStructs with different modes or mode_params! Got modes %s and %s with mode_params %s and %s"%(self._mode, other._mode, self._mode_params, other._mode_params))
        
        new_data = ProfileStruct(
            xvar = np.concatenate([self.data.xvar, other.data.xvar]),
            yvar = np.concatenate([self.data.yvar, other.data.yvar]),
            mode = self._mode, # pyright: ignore[reportArgumentType]
            mode_params = self._mode_params
        )

        return ProfileHistStruct(
            data=new_data,
            axes=self.axes,
         )
    
    '''
    Mimick the hist.Hist() interface
    '''

    #statistic
    def values(self, flow=False):
        return self._statistic
    
    #just return 0 for uncertainties atm
    def variances(self, flow=False):
        return np.zeros_like(self._statistic)