from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.patches import Circle as MplCircle

from rtnls_enface.base import Point

from .base import Grid, GridField, GridFieldEnum

if TYPE_CHECKING:
    from rtnls_enface.fundus import Fundus


class ODOnlyField(GridFieldEnum):
    """Fields of ODOnlyGrid."""

    Superior = "superior", "superior region"
    Inferior = "inferior", "inferior region"
    Left = "left", "left region"
    Right = "right", "right region"
    FullGrid = "grid", "full region"

    @staticmethod
    def grid():
        return ODOnlyGrid


class ODOnlyGrid(Grid):
    """Fixed-radius circle centered on the optic disc.

    Size is specified in pixels and does not depend on the fovea. Superior/inferior
    are split by a horizontal line through the disc center; left/right by a
    vertical line through the disc center (image axes).
    """

    def __init__(
        self,
        retina: Fundus,
        resolution,
        mask,
        radius_px: float = 400.0,
    ):
        super().__init__(retina, resolution, mask)
        if retina.disc is None:
            raise ValueError("ODOnlyGrid requires an optic disc")
        radius_px = float(radius_px)
        if not np.isfinite(radius_px) or radius_px <= 0:
            raise ValueError("radius_px must be a positive finite value")
        self.retina = retina
        self.radius_px = radius_px
        center = retina.disc.circle.center
        self._center = Point(center.y, center.x)

    def _full_field_bounds(self, field: GridFieldEnum):
        return self._circle_bounds(self._center.x, self._center.y, self.radius_px)

    def _shift_canvas_origin(self, x_offset: int, y_offset: int) -> None:
        self._center = Point(
            self._center.y - y_offset, self._center.x - x_offset
        )

    @cached_property
    def _dx(self) -> np.ndarray:
        return np.arange(self.w, dtype=float)[None, :] - self._center.x

    @cached_property
    def _dy(self) -> np.ndarray:
        return np.arange(self.h, dtype=float)[:, None] - self._center.y

    @cached_property
    def grid(self) -> np.ndarray:
        """Boolean mask of the full disc-centered circle."""
        r2 = self.radius_px * self.radius_px
        return (self._dx * self._dx) + (self._dy * self._dy) <= r2

    @cached_property
    def superior(self) -> np.ndarray:
        """Upper half of the circle (smaller image y)."""
        return self.grid & (self._dy < 0)

    @cached_property
    def inferior(self) -> np.ndarray:
        """Lower half of the circle (larger image y)."""
        return self.grid & ~(self._dy < 0)

    @cached_property
    def left(self) -> np.ndarray:
        """Left half of the circle (smaller image x)."""
        return self.grid & (self._dx < 0)

    @cached_property
    def right(self) -> np.ndarray:
        """Right half of the circle (larger image x)."""
        return self.grid & ~(self._dx < 0)

    def plot(self, ax, field: ODOnlyField | GridField | None = None):
        """Plot the circle, axis-aligned dividers, and optionally highlight a field."""
        cx, cy, r = self._center.x, self._center.y, self.radius_px
        ax.add_artist(
            MplCircle(
                (cx, cy),
                radius=r,
                fill=False,
                edgecolor="white",
                linewidth=0.5,
            )
        )

        field_enum = field.enum if isinstance(field, GridField) else field
        draw_horizontal = field_enum in (
            None,
            ODOnlyField.FullGrid,
            ODOnlyField.Superior,
            ODOnlyField.Inferior,
        )
        draw_vertical = field_enum in (
            None,
            ODOnlyField.FullGrid,
            ODOnlyField.Left,
            ODOnlyField.Right,
        )
        if draw_horizontal:
            ax.plot([cx - r, cx + r], [cy, cy], color="white", linewidth=0.5)
        if draw_vertical:
            ax.plot([cx, cx], [cy - r, cy + r], color="white", linewidth=0.5)

        if field is not None:
            mask = getattr(self, field_enum.value)
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

    FieldEnum = ODOnlyField
