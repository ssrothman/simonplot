from typing import List, Sequence, Never, Protocol, Any, Union, Sequence, Tuple, runtime_checkable
import awkward as ak
import matplotlib.axes
import numpy as np
import hist
from enum import IntEnum
from typing import overload

from simon_mpl_util.util.AribtraryBinning import ArbitraryBinning

@runtime_checkable
class HasKeyProtocol(Protocol):
    @property
    def key(self) -> str:
        ...

@runtime_checkable
class VariableLikeProtocol(HasKeyProtocol, Protocol):
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

@runtime_checkable
class CutProtocol(VariableLikeProtocol, Protocol):   
    '''
    The type logic here is more complicated than I can get the static type checking to understand
    So I just leave it as Any :(

    The actual type logic is:
    NoCut().evaluate(Any) -> slice
    CutProtocol.evaluate(UnbinnedDatasetAccessProtocol) -> np.ndarray[bool] | ak.Array[bool]
    CutProtocol.evaluate(PrebinnedDatasetAccessProtocol) -> np.ndarray[float] | Tuple[np.ndarray[float], np.ndarray[float]]
    '''
    def evaluate(self, dataset : "UnbinnedDatasetAccessProtocol | PrebinnedDatasetAccessProtocol") -> Any:
        ...

@runtime_checkable
class PrebinnedOperationProtocol(CutProtocol, Protocol):
    def resulting_binning(self, binning : ArbitraryBinning) -> ArbitraryBinning:
        ...

    def _compute_resulting_binning(self, binning : ArbitraryBinning) -> ArbitraryBinning:
        ...

@runtime_checkable
class VariableProtocol(VariableLikeProtocol, Protocol):
    #variable return type is more permissive
    def evaluate(self, dataset : "UnbinnedDatasetAccessProtocol | PrebinnedDatasetAccessProtocol", cut : CutProtocol) -> Any:
        ...

    @property
    def centerline(self) -> None | float | Sequence[float]:
        ...

class BinningKind(IntEnum):
    AUTO = 0
    DEFAULT = 1
    BASIC = 2
    PREBINNED = 3

@runtime_checkable
class BaseBinningProtocol(Protocol):
    @property
    def kind(self) -> BinningKind:
        ...

    @property
    def has_custom_labels(self) -> bool:
        ...
    
    @property
    def label_lookup(self) -> dict[str, str]:
        ...

@runtime_checkable
class AutoBinningProtocol(BaseBinningProtocol, Protocol):
    def build_auto_axis(self, 
                        variables: Sequence[VariableProtocol], 
                        cuts: Sequence[CutProtocol], 
                        datasets: Sequence["BaseDatasetProtocol"], 
                        transform: Union[str, None]=None) -> hist.axis.AxesMixin:
        ...

@runtime_checkable 
class DefaultBinningProtocol(BaseBinningProtocol, Protocol):
    def build_default_axis(self, variable: VariableProtocol) -> hist.axis.AxesMixin:
        ...

@runtime_checkable
class BasicBinningProtocol(BaseBinningProtocol, Protocol):
    def build_axis(self, variable : VariableProtocol) -> hist.axis.AxesMixin:
        ...
    
@runtime_checkable
class PrebinnedBinningProtocol(BaseBinningProtocol, Protocol):
    def build_prebinned_axis(self, 
                             dataset : "BaseDatasetProtocol",
                             cut : PrebinnedOperationProtocol) -> ArbitraryBinning:
        ...

type AnyBinningProtocol = AutoBinningProtocol | DefaultBinningProtocol | BasicBinningProtocol | PrebinnedBinningProtocol

class BaseDatasetProtocol(Protocol):
    @property
    def is_stack(self) -> bool:
        ...

    def set_label(self, label : str) -> None:
        ...

    def set_color(self, color : Any) -> None:
        ...

    @property
    def label(self) -> str | None:
        ...

    @property
    def key(self) -> str:
        ...

    @property
    def color(self) -> Any | None:
        ...

    def set_lumi(self, lumi: float) -> None:
        ...

    def set_xsec(self, xsec: float) -> None:
        ...

    def override_num_events(self, nevts: float) -> None:
        ...

    @property
    def num_events(self) -> float:
        ...

    @property
    def lumi(self) -> float:
        ...
    
    @property
    def xsec(self) -> float:
        ...
    
    @property
    def isMC(self) -> bool:
        ...
    
    def compute_weight(self, target_lumi : float) -> None:
        ...

    def fill_hist(self,
                  variable: VariableProtocol, 
                  cut: CutProtocol, 
                  weight : VariableProtocol,
                  axis : Any) -> Any:
        ...

    def plot_hist(self,
                       variable: VariableProtocol, 
                       cut: CutProtocol, 
                       weight : VariableProtocol,
                       axis : Any,
                       density: bool,
                       ax : matplotlib.axes.Axes,
                       own_style : bool,
                       fillbetween : Union[float, None],
                       **mpl_kwargs) -> Tuple[Any, Any]:
        ...

    def plot_hist_ratio(self,
                    H1 : Any,
                    H2 : Any,
                    axis : Any,
                    density : bool,
                    ax : matplotlib.axes.Axes,
                    own_style : bool,
                    **mpl_kwargs):
        ...

@runtime_checkable
class UnbinnedDatasetAccessProtocol(Protocol):
    def ensure_columns(self, columns : Sequence[str]) -> None:
        ...

    def get_column(self, column_name: str, collection_name: str | None) -> np.ndarray | ak.Array:
        ...
            
    @property
    def num_rows(self) -> int:
        ...

@runtime_checkable
class PrebinnedDatasetAccessProtocol(Protocol):
    @property
    def data(self) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        ...

    @property
    def binning(self) -> ArbitraryBinning:
        ...

    def project(self, axes : Sequence[str]):
        raise NotImplementedError()

    def slice(self, edges):
        raise NotImplementedError()

    def _dummy_dset(self, data : np.ndarray | Tuple[np.ndarray, np.ndarray], binning : ArbitraryBinning) -> "PrebinnedDatasetAccessProtocol":
        ...

class UnbinnedDatasetProtocol(BaseDatasetProtocol, UnbinnedDatasetAccessProtocol, Protocol):
    pass

class PrebinnedDatasetProtocol(BaseDatasetProtocol, PrebinnedDatasetAccessProtocol, Protocol):
    @property
    def quantitytype(self) -> str:
        ...

type AnyDatasetProtocol = PrebinnedDatasetAccessProtocol | UnbinnedDatasetAccessProtocol