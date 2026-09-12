from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
from typing import ClassVar, Generic, Optional, Type, TypeVar

from rtnls_enface.grids.base import Grid, GridFieldEnum
from rtnls_enface.grids.circle import CircleGrid
from rtnls_enface.grids.disc_centered import DiscCenteredGrid
from rtnls_enface.grids.ellipse import EllipseGrid
from rtnls_enface.grids.etdrs import ETDRSGrid
from rtnls_enface.grids.hemifields import HemifieldGrid
from rtnls_enface.grids.od_only import ODOnlyGrid

GridT = TypeVar("GridT", bound=Grid)
FieldEnumT = TypeVar("FieldEnumT", bound=GridFieldEnum)


def _validate_min_area_within_bounds(value: Optional[float]) -> None:
    """Validate an optional fraction of a grid field that must be visible."""
    if value is None:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError("min_area_within_bounds must be a finite fraction in [0.0, 1.0]")


@dataclass(frozen=True)
class BaseGridSpecification(ABC, Generic[GridT]):
    """Base class for immutable grid hyper-parameter specifications."""

    # The compact name is an instance-level override so every grid
    # specification can use it in feature names. Concrete specifications
    # provide their normal value through ``default_name`` below.
    name: str = ""
    description: str = field(default="", compare=False, hash=False)
    default_name: ClassVar[str] = ""
    display_name: ClassVar[str] = ""
    default_description: ClassVar[str] = "a retinal measurement grid"
    requires_fovea: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if not self.name:
            object.__setattr__(self, "name", self.default_name)
        if not self.description:
            object.__setattr__(self, "description", self.default_description)
        _validate_min_area_within_bounds(
            getattr(self, "min_area_within_bounds", None)
        )

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
    default_description: ClassVar[str] = (
        "a fovea-centered ETDRS grid with ring radii of 0.5, 1.5, and 3.0 mm"
    )
    multiplier: float = 1.0
    min_area_within_bounds: Optional[float] = None

    @property
    def grid_cls(self) -> Type[ETDRSGrid]:
        return ETDRSGrid

    def init_kwargs(self) -> dict[str, object]:
        return {"multiplier": self.multiplier}


@dataclass(frozen=True)
class CircleGridSpecification(BaseGridSpecification[CircleGrid]):
    """Hyper-parameters for OD–fovea circle grids.

    ``center`` places the circle along the OD–fovea axis (0 = disc, 1 = fovea).
    ``radius_multiplier`` scales the OD–fovea distance term of the radius
    (``radius = radius_multiplier * d_ODF + disc radius``). ``multiplier`` is a
    deprecated alias for ``center``.
    ``name`` is an optional compact token override used when serializing grid
    specifications into feature names.
    """

    default_name: ClassVar[str] = "crcl"
    display_name: ClassVar[str] = "Circle"
    default_description: ClassVar[str] = (
        "a circular grid positioned along the optic-disc–fovea axis and scaled "
        "to the distance between these landmarks"
    )
    center: float | None = None
    radius_multiplier: float = 1.0
    band_crop_fraction: float = 0.0
    multiplier: float | None = None  # deprecated alias for center
    min_area_within_bounds: Optional[float] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.center is not None and self.multiplier is not None:
            raise ValueError(
                "Pass only one of 'center' or 'multiplier'; "
                "'multiplier' is a deprecated alias for 'center'"
            )
        resolved_center = (
            1.0
            if self.center is None and self.multiplier is None
            else float(self.center if self.center is not None else self.multiplier)
        )
        object.__setattr__(self, "center", resolved_center)
        object.__setattr__(self, "multiplier", None)

    @property
    def grid_cls(self) -> Type[CircleGrid]:
        return CircleGrid

    def init_kwargs(self) -> dict[str, object]:
        return {
            "center": self.center,
            "radius_multiplier": self.radius_multiplier,
            "band_crop_fraction": self.band_crop_fraction,
        }


@dataclass(frozen=True)
class EllipseGridSpecification(BaseGridSpecification[EllipseGrid]):
    """Hyper-parameters for OD–fovea ellipse grids."""

    default_name: ClassVar[str] = "elps"
    display_name: ClassVar[str] = "Ellipse"
    default_description: ClassVar[str] = (
        "an ellipse centered midway between the optic disc and fovea, aligned "
        "with their axis and scaled to their separation"
    )
    multiplier: float = 1.0
    band_crop_fraction: float = 0.0
    min_area_within_bounds: Optional[float] = None

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
    default_description: ClassVar[str] = (
        "an optic-disc-centered annulus scaled to the optic-disc–fovea distance"
    )
    multiplier: float = 1.0
    band_crop_fraction: float = 0.0
    min_area_within_bounds: Optional[float] = None

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
    default_description: ClassVar[str] = (
        "the visible retina divided into superior and inferior regions by the "
        "optic-disc–fovea axis"
    )
    min_area_within_bounds: Optional[float] = None

    @property
    def grid_cls(self) -> Type[HemifieldGrid]:
        return HemifieldGrid


@dataclass(frozen=True)
class ODOnlyGridSpecification(BaseGridSpecification[ODOnlyGrid]):
    """Hyper-parameters for fixed-pixel optic-disc–centered grids."""

    default_name: ClassVar[str] = "odonly"
    display_name: ClassVar[str] = "OD-only"
    default_description: ClassVar[str] = (
        "a fixed-radius circular grid centered on the optic disc without "
        "foveal localization"
    )
    requires_fovea: ClassVar[bool] = False
    radius_px: float = 400.0
    min_area_within_bounds: Optional[float] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        radius_px = float(self.radius_px)
        if not isfinite(radius_px) or radius_px <= 0:
            raise ValueError("radius_px must be a positive finite value")
        object.__setattr__(self, "radius_px", radius_px)

    @property
    def grid_cls(self) -> Type[ODOnlyGrid]:
        return ODOnlyGrid

    def init_kwargs(self) -> dict[str, object]:
        return {"radius_px": self.radius_px}


@dataclass(frozen=True)
class BaseGridFieldSpecification(Generic[FieldEnumT]):
    """Base specification bundling a grid spec with a field enum."""

    grid_spec: BaseGridSpecification
    field: FieldEnumT
    min_area_within_bounds: Optional[float] = None

    def __post_init__(self) -> None:
        _validate_min_area_within_bounds(self.min_area_within_bounds)

    @property
    def description(self) -> str:
        """Return a publication-facing description of this region of interest."""
        return f"the {self.field.friendly_name} of {self.grid_spec.description}"


@dataclass(frozen=True)
class GridFieldSpecification(
    BaseGridFieldSpecification[FieldEnumT], Generic[FieldEnumT]
):
    """Concrete dataclass for specifying a grid field within a grid spec."""
