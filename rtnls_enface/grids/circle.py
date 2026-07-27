from __future__ import annotations

import math
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.patches import Circle as MplCircle

from rtnls_enface.base import Line, Point

from .base import Grid, GridField, GridFieldEnum

if TYPE_CHECKING:
    from rtnls_enface.fundus import Fundus


class CircleField(GridFieldEnum):
    """Fields of CircleGrid."""

    Superior = "superior"
    Inferior = "inferior"
    FullGrid = "grid"

    @staticmethod
    def grid():
        return CircleGrid


class CircleGrid(Grid):
    """Circle-based grid centered along the OD-fovea axis.

    `center` controls the center position from optic disc (0.0) to fovea (1.0).
    The radius is `radius_multiplier * OD-fovea distance + optic disc radius`.
    `multiplier` is a deprecated alias for `center`.
    Optionally crops bands parallel to the OD-fovea axis from both sides.
    Exposes boolean masks for the full circle (grid) and its superior/inferior hemifields
    split by a line through the circle center perpendicular to the OD–fovea axis.
    """

    def __init__(
        self,
        retina: Fundus,
        resolution,
        mask,
        multiplier: float | None = None,
        band_crop_fraction: float = 0.0,
        center: float | None = None,
        radius_multiplier: float = 1.0,
    ):
        super().__init__(retina, resolution, mask)
        self.retina = retina
        self.center = self._resolve_center(center=center, multiplier=multiplier)
        self.multiplier = self.center  # deprecated alias for center
        self.radius_multiplier = float(radius_multiplier)
        self.band_crop_fraction = self._validate_band_crop_fraction(band_crop_fraction)

        od_to_fovea = Line(retina.disc.center_of_mass, retina.fovea_location)
        self._center = od_to_fovea.point_at(self.center)
        self._r = (
            self.radius_multiplier * od_to_fovea.length + retina.disc.circle.r
        )
        self._angle_deg = float(od_to_fovea.orientation())

    def _full_field_bounds(self, field: GridFieldEnum):
        return self._circle_bounds(self._center.x, self._center.y, self._r)

    def _shift_canvas_origin(self, x_offset: int, y_offset: int) -> None:
        self._center = Point(
            self._center.y - y_offset, self._center.x - x_offset
        )

    @staticmethod
    def _resolve_center(
        center: float | None, multiplier: float | None
    ) -> float:
        """Resolve center from `center` or deprecated `multiplier` alias."""
        if center is not None and multiplier is not None:
            raise ValueError(
                "Pass only one of 'center' or 'multiplier'; "
                "'multiplier' is a deprecated alias for 'center'"
            )
        if center is None and multiplier is None:
            return 1.0
        return float(center if center is not None else multiplier)

    @staticmethod
    def _validate_band_crop_fraction(value: float) -> float:
        """Validate the fraction cropped from each side of the circle."""
        value = float(value)
        if not 0.0 <= value < 0.5:
            raise ValueError("band_crop_fraction must be in the range [0.0, 0.5)")
        return value

    @cached_property
    def _theta(self) -> float:
        theta = math.radians(self._angle_deg)
        # Normalize so the OD–fovea direction runs left→right (cos(theta) >= 0)
        if math.cos(theta) < 0.0:
            theta += math.pi
        return theta

    @cached_property
    def _cos_sin(self) -> tuple[float, float]:
        return math.cos(self._theta), math.sin(self._theta)

    @cached_property
    def _dx(self) -> np.ndarray:
        cx = self._center.x
        return np.arange(self.w, dtype=float)[None, :] - cx

    @cached_property
    def _dy(self) -> np.ndarray:
        cy = self._center.y
        return np.arange(self.h, dtype=float)[:, None] - cy

    @cached_property
    def grid(self) -> np.ndarray:
        """Boolean mask of the full circle."""
        r2 = self._r * self._r
        inside = (self._dx * self._dx) + (self._dy * self._dy) <= r2
        return inside & self._band_crop_mask

    @cached_property
    def _perp_unit(self) -> tuple[float, float]:
        # Unit vector along OD–fovea axis in image coords
        c, s = self._cos_sin
        # Perpendicular vector (rotate by +90deg): (-s, c)
        px, py = -s, c
        norm = math.hypot(px, py)
        return px / norm, py / norm

    @cached_property
    def _perp_coord(self) -> np.ndarray:
        """Signed pixel distance perpendicular to the OD-fovea axis."""
        px, py = self._perp_unit
        return self._dx * px + self._dy * py

    @cached_property
    def _band_crop_mask(self) -> np.ndarray:
        """Mask that removes bands parallel to the OD-fovea axis."""
        if self.band_crop_fraction == 0.0:
            return np.ones((self.h, self.w), dtype=bool)

        allowed_half_width = self._r * (1.0 - 2.0 * self.band_crop_fraction)
        return np.abs(self._perp_coord) <= allowed_half_width

    @cached_property
    def superior(self) -> np.ndarray:
        """Boolean mask of the superior hemifield of the circle."""
        px, py = self._perp_unit
        # Dot < 0 corresponds to superior (y smaller) region in image coords
        dot = self._dx * px + self._dy * py
        return self.grid & (dot < 0)

    @cached_property
    def inferior(self) -> np.ndarray:
        """Boolean mask of the inferior hemifield of the circle."""
        px, py = self._perp_unit
        dot = self._dx * px + self._dy * py
        return self.grid & ~(dot < 0)

    def plot(self, ax, field: CircleField | GridField | None = None):
        """Plot the circle outline, divider, and optionally highlight a field."""
        if self.band_crop_fraction == 0.0:
            patch = MplCircle(
                (self._center.x, self._center.y),
                radius=self._r,
                fill=False,
                edgecolor="white",
                linewidth=0.5,
            )
            ax.add_artist(patch)
        else:
            ax.contour(
                self.grid.astype(float),
                levels=[0.5],
                colors="white",
                linewidths=0.5,
            )

        # Divider line (parallel to OD–fovea), clipped to the circle interior (diameter)
        c, s = self._cos_sin
        x1 = self._center.x - self._r * c
        y1 = self._center.y - self._r * s
        x2 = self._center.x + self._r * c
        y2 = self._center.y + self._r * s
        ax.plot([x1, x2], [y1, y2], color="white", linestyle="-", linewidth=0.5)

        # Highlight a specific field
        if field is not None:
            enum = field.enum if isinstance(field, GridField) else field
            mask = getattr(self, enum.value)
            highlighted = np.ma.masked_where(~mask, np.ones_like(mask))
            ax.imshow(
                highlighted,
                cmap="Greys_r",
                alpha=0.2,
                extent=[0, self.w, self.h, 0],
            )

        ax.set_xlim(0, self.w)
        ax.set_ylim(self.h, 0)
        ax.axis("off")

    # Link enum type to grid
    FieldEnum = CircleField
