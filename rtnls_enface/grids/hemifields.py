from __future__ import annotations

import math
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from rtnls_enface.base import Line
from .base import Grid, GridFieldEnum, GridField

if TYPE_CHECKING:
    from rtnls_enface.fundus import Fundus


class HemifieldField(GridFieldEnum):
    """Fields of HemifieldGrid."""

    Superior = "superior"
    Inferior = "inferior"
    FullGrid = "grid"

    @staticmethod
    def grid():
        return HemifieldGrid


class HemifieldGrid(Grid):
    """Simple grid that splits the image into superior/inferior halves.

    The split is defined relative to the optic disc–fovea axis. Masks cover the
    entire image half-planes (no radial cutoff). The plot draws only the axis line.
    """

    def __init__(self, retina: Fundus, resolution, mask):
        super().__init__(retina, resolution, mask)
        self.retina = retina

        # Geometry from OD–fovea
        od_to_fovea = Line(retina.disc.center_of_mass, retina.fovea_location)
        self._center = od_to_fovea.point_at(0.5)
        self._angle_deg = float(od_to_fovea.orientation())

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
        """Boolean mask of the full image (all ones)."""
        return np.ones((self.h, self.w), dtype=bool)

    @cached_property
    def _perp_unit(self) -> tuple[float, float]:
        # Unit vector along OD–fovea axis in image coords
        c, s = self._cos_sin
        # Perpendicular vector (rotate by +90deg): (-s, c)
        px, py = -s, c
        norm = math.hypot(px, py)
        return px / norm, py / norm

    @cached_property
    def superior(self) -> np.ndarray:
        """Boolean mask of the superior hemifield of the image."""
        px, py = self._perp_unit
        # Dot < 0 corresponds to superior (y smaller) region in image coords
        dot = self._dx * px + self._dy * py
        return dot < 0

    @cached_property
    def inferior(self) -> np.ndarray:
        """Boolean mask of the inferior hemifield of the image."""
        px, py = self._perp_unit
        dot = self._dx * px + self._dy * py
        return ~(dot < 0)

    def plot(self, ax, field: HemifieldField | GridField | None = None):
        """Plot only the OD–fovea axis line and optionally highlight a field."""
        # Axis line across the image extents
        c, s = self._cos_sin
        length = max(self.w, self.h) * 2.0
        x1 = self._center.x - length * c
        y1 = self._center.y - length * s
        x2 = self._center.x + length * c
        y2 = self._center.y + length * s
        ax.plot([x1, x2], [y1, y2], color="white", linestyle="-", linewidth=0.5)

        # Highlight a specific field
        if field is not None:
            enum = field.enum if isinstance(field, GridField) else field
            mask = getattr(self, enum.value)
            highlighted = np.ma.masked_where(~mask, np.ones_like(mask))
            ax.imshow(
                highlighted,
                cmap="gray",
                alpha=0.2,
                extent=[0, self.w, self.h, 0],
            )

        ax.set_xlim(0, self.w)
        ax.set_ylim(self.h, 0)
        ax.axis("off")

    # Link enum type to grid
    FieldEnum = HemifieldField


