from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Generic, Type, TypeVar

from rtnls_enface.grids.base import Grid, GridFieldEnum
from rtnls_enface.grids.circle import CircleGrid
from rtnls_enface.grids.disc_centered import DiscCenteredGrid
from rtnls_enface.grids.ellipse import EllipseGrid
from rtnls_enface.grids.etdrs import ETDRSGrid
from rtnls_enface.grids.hemifields import HemifieldGrid

GridT = TypeVar("GridT", bound=Grid)
FieldEnumT = TypeVar("FieldEnumT", bound=GridFieldEnum)


@dataclass(frozen=True)
class BaseGridSpecification(ABC, Generic[GridT]):
    """Base class for immutable grid hyper-parameter specifications."""

    # The compact name is an instance-level override so every grid
    # specification can use it in feature names. Concrete specifications
    # provide their normal value through ``default_name`` below.
    name: str = ""
    default_name: ClassVar[str] = ""
    display_name: ClassVar[str] = ""

    def __post_init__(self) -> None:
        if not self.name:
            object.__setattr__(self, "name", self.default_name)

    @property
    @abstractmethod
    def grid_cls(self) -> Type[GridT]:
        """Return the `Grid` subclass this specification instantiates."""
        raise NotImplementedError

    def init_kwargs(self) -> dict[str, object]:
        """Additional keyword arguments to forward to the grid constructor."""
        return {}


@dataclass(frozen=True)
class ETDRSGridSpecification(BaseGridSpecification[ETDRSGrid]):
    """Hyper-parameters for ETDRS grids."""

    default_name: ClassVar[str] = "etdrs"
    display_name: ClassVar[str] = "ETDRS"
    multiplier: float = 1.0

    @property
    def grid_cls(self) -> Type[ETDRSGrid]:
        return ETDRSGrid

    def init_kwargs(self) -> dict[str, object]:
        return {"multiplier": self.multiplier}


@dataclass(frozen=True)
class CircleGridSpecification(BaseGridSpecification[CircleGrid]):
    """Hyper-parameters for OD–fovea circle grids.

    ``name`` is an optional compact token override used when serializing grid
    specifications into feature names. The option is defined by the base
    specification and is available consistently on every grid type.
    """

    default_name: ClassVar[str] = "crcl"
    display_name: ClassVar[str] = "Circle"
    multiplier: float = 1.0
    band_crop_fraction: float = 0.0

    @property
    def grid_cls(self) -> Type[CircleGrid]:
        return CircleGrid

    def init_kwargs(self) -> dict[str, object]:
        return {
            "multiplier": self.multiplier,
            "band_crop_fraction": self.band_crop_fraction,
        }


@dataclass(frozen=True)
class EllipseGridSpecification(BaseGridSpecification[EllipseGrid]):
    """Hyper-parameters for OD–fovea ellipse grids."""

    default_name: ClassVar[str] = "elps"
    display_name: ClassVar[str] = "Ellipse"
    multiplier: float = 1.0
    band_crop_fraction: float = 0.0

    @property
    def grid_cls(self) -> Type[EllipseGrid]:
        return EllipseGrid

    def init_kwargs(self) -> dict[str, object]:
        return {
            "multiplier": self.multiplier,
            "band_crop_fraction": self.band_crop_fraction,
        }


@dataclass(frozen=True)
class DiscCenteredGridSpecification(BaseGridSpecification[DiscCenteredGrid]):
    """Hyper-parameters for disc-centered grids."""

    default_name: ClassVar[str] = "disc"
    display_name: ClassVar[str] = "Disc"
    multiplier: float = 1.0
    band_crop_fraction: float = 0.0

    @property
    def grid_cls(self) -> Type[DiscCenteredGrid]:
        return DiscCenteredGrid

    def init_kwargs(self) -> dict[str, object]:
        return {
            "multiplier": self.multiplier,
            "band_crop_fraction": self.band_crop_fraction,
        }


@dataclass(frozen=True)
class HemifieldGridSpecification(BaseGridSpecification[HemifieldGrid]):
    """Hyper-parameters for hemifield grids (currently parameter-free)."""

    default_name: ClassVar[str] = "hf"
    display_name: ClassVar[str] = "Hemifield"

    @property
    def grid_cls(self) -> Type[HemifieldGrid]:
        return HemifieldGrid


@dataclass(frozen=True)
class BaseGridFieldSpecification(Generic[FieldEnumT]):
    """Base specification bundling a grid spec with a field enum."""

    grid_spec: BaseGridSpecification
    field: FieldEnumT


@dataclass(frozen=True)
class GridFieldSpecification(
    BaseGridFieldSpecification[FieldEnumT], Generic[FieldEnumT]
):
    """Concrete dataclass for specifying a grid field within a grid spec."""
