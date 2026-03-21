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


class BaseGridSpecification(ABC, Generic[GridT]):
    """Base class for immutable grid hyper-parameter specifications."""

    name: ClassVar[str]
    display_name: ClassVar[str] = ""

    @property
    @abstractmethod
    def grid_cls(self) -> Type[GridT]:
        """Return the `Grid` subclass this specification instantiates."""
        raise NotImplementedError

    def init_kwargs(self) -> dict[str, object]:
        """Additional keyword arguments to forward to the grid constructor."""
        return {}


@dataclass(frozen=True, slots=True)
class ETDRSGridSpecification(BaseGridSpecification[ETDRSGrid]):
    """Hyper-parameters for ETDRS grids."""

    name: ClassVar[str] = "etdrs"
    display_name: ClassVar[str] = "ETDRS"
    multiplier: float = 1.0

    @property
    def grid_cls(self) -> Type[ETDRSGrid]:
        return ETDRSGrid

    def init_kwargs(self) -> dict[str, object]:
        return {"multiplier": self.multiplier}


@dataclass(frozen=True, slots=True)
class CircleGridSpecification(BaseGridSpecification[CircleGrid]):
    """Hyper-parameters for OD–fovea circle grids."""

    name: ClassVar[str] = "crcl"
    display_name: ClassVar[str] = "Circle"
    multiplier: float = 1.0

    @property
    def grid_cls(self) -> Type[CircleGrid]:
        return CircleGrid

    def init_kwargs(self) -> dict[str, object]:
        return {"multiplier": self.multiplier}


@dataclass(frozen=True, slots=True)
class EllipseGridSpecification(BaseGridSpecification[EllipseGrid]):
    """Hyper-parameters for OD–fovea ellipse grids."""

    name: ClassVar[str] = "elps"
    display_name: ClassVar[str] = "Ellipse"
    multiplier: float = 1.0

    @property
    def grid_cls(self) -> Type[EllipseGrid]:
        return EllipseGrid

    def init_kwargs(self) -> dict[str, object]:
        return {"multiplier": self.multiplier}


@dataclass(frozen=True, slots=True)
class DiscCenteredGridSpecification(BaseGridSpecification[DiscCenteredGrid]):
    """Hyper-parameters for disc-centered grids."""

    name: ClassVar[str] = "disc"
    display_name: ClassVar[str] = "Disc"
    multiplier: float = 1.0

    @property
    def grid_cls(self) -> Type[DiscCenteredGrid]:
        return DiscCenteredGrid

    def init_kwargs(self) -> dict[str, object]:
        return {"multiplier": self.multiplier}


@dataclass(frozen=True, slots=True)
class HemifieldGridSpecification(BaseGridSpecification[HemifieldGrid]):
    """Hyper-parameters for hemifield grids (currently parameter-free)."""

    name: ClassVar[str] = "hf"
    display_name: ClassVar[str] = "Hemifield"

    @property
    def grid_cls(self) -> Type[HemifieldGrid]:
        return HemifieldGrid


@dataclass(frozen=True, slots=True)
class BaseGridFieldSpecification(Generic[FieldEnumT]):
    """Base specification bundling a grid spec with a field enum."""

    grid_spec: BaseGridSpecification
    field: FieldEnumT


@dataclass(frozen=True, slots=True)
class GridFieldSpecification(
    BaseGridFieldSpecification[FieldEnumT], Generic[FieldEnumT]
):
    """Concrete dataclass for specifying a grid field within a grid spec."""
