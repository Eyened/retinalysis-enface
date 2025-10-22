from __future__ import annotations

import math
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.patches import Ellipse as MplEllipse

from rtnls_enface.base import Line
from .base import Grid, GridFieldEnum, GridField

if TYPE_CHECKING:
    from rtnls_enface.fundus import Fundus


class EllipseField(GridFieldEnum):
    """Fields of EllipseGrid."""

    Superior = "superior"
    Inferior = "inferior"

    @staticmethod
    def grid():
        return EllipseGrid


class EllipseGrid(Grid):
    """Ellipse-based grid centered midway between OD and fovea.

    The ellipse has semi-axes a=radius and b=1.5*radius, rotated to the
    OD–fovea axis. The grid exposes boolean masks for the full ellipse
    (grid) and its superior/inferior hemifields split by a line through
    the ellipse center perpendicular to the OD–fovea axis.
    """

    def __init__(self, retina: Fundus, resolution, mask, multiplier: float = 1.0):
        super().__init__(retina, resolution, mask)
        self.retina = retina
        self.multiplier = multiplier

        # Geometry from OD–fovea
        od_to_fovea = Line(retina.disc.center_of_mass, retina.fovea_location)
        self._center = od_to_fovea.point_at(0.5)
        base_radius = od_to_fovea.length / 2 + retina.disc.circle.r
        self._a = float(base_radius) * float(self.multiplier)
        self._b = 1.5 * float(base_radius) * float(self.multiplier)
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
    def _xprime_yprime(self) -> tuple[np.ndarray, np.ndarray]:
        c, s = self._cos_sin
        # Rotate coordinate system so that ellipse equation is axis-aligned
        xprime = self._dx * c + self._dy * s
        yprime = -self._dx * s + self._dy * c
        return xprime, yprime

    @cached_property
    def grid(self) -> np.ndarray:
        """Boolean mask of the full ellipse."""
        xprime, yprime = self._xprime_yprime
        a2 = (self._a * self._a)
        b2 = (self._b * self._b)
        inside = (xprime * xprime) / a2 + (yprime * yprime) / b2 <= 1.0
        return inside

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
        """Boolean mask of the superior hemifield of the ellipse."""
        px, py = self._perp_unit
        # Dot < 0 corresponds to superior (y smaller) region in image coords
        dot = self._dx * px + self._dy * py
        return self.grid & (dot < 0)

    @cached_property
    def inferior(self) -> np.ndarray:
        """Boolean mask of the inferior hemifield of the ellipse."""
        px, py = self._perp_unit
        dot = self._dx * px + self._dy * py
        return self.grid & ~(dot < 0)

    def plot(self, ax, field: EllipseField | GridField | None = None):
        """Plot the ellipse outline, divider, and optionally highlight a field."""
        # Outline
        patch = MplEllipse(
            (self._center.x, self._center.y),
            width=2 * self._a,
            height=2 * self._b,
            angle=self._angle_deg,
            fill=False,
            edgecolor="white",
            linewidth=1,
        )
        ax.add_artist(patch)

        # Divider line (parallel to OD–fovea), clipped to the ellipse interior (±a)
        c, s = self._cos_sin
        x1 = self._center.x - self._a * c
        y1 = self._center.y - self._a * s
        x2 = self._center.x + self._a * c
        y2 = self._center.y + self._a * s
        ax.plot([x1, x2], [y1, y2], color="white", linestyle="-", linewidth=1)

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
    FieldEnum = EllipseField


