from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.patches import Circle as MplCircle
from skimage import measure

from rtnls_enface.base import Laterality

from .base import Grid, GridFieldEnum, GridField

if TYPE_CHECKING:
    from rtnls_enface.fundus import Fundus


class DiscCenteredField(GridFieldEnum):
    # FullGrid = "grid"
    @staticmethod
    def grid():
        return DiscCenteredGrid


class DiscCenteredRing(DiscCenteredField):
    FullGrid = "grid"
    Center = "center"
    Inner = "inner"
    Outer = "outer"


class DiscCenteredQuadrant(DiscCenteredField):
    Superior = "quadrant_superior"
    Inferior = "quadrant_inferior"
    Nasal = "quadrant_nasal"
    Temporal = "quadrant_temporal"
    Right = "quadrant_right"
    Left = "quadrant_left"


class DiscCenteredGrid(Grid):
    def __init__(self, retina: Fundus, resolution, mask):
        super().__init__(retina, resolution, mask)
        # Anchor at disc center
        self.disc_cx = retina.disc.circle.center.x
        self.disc_cy = retina.disc.circle.center.y
        self.laterality = retina.laterality
        # Physical distances (mm)
        self._odfd = self.resolution * retina.disc_fovea_distance
        self._disc_radius_mm = self.resolution * retina.disc.circle.r

    def calculate_area(self, binary_image):
        return binary_image.sum() * self.resolution**2

    def calculate_count(self, binary_image):
        return np.max(measure.label(binary_image))

    def get_summary(self, binary_image, fields, include_area=True, include_count=True):
        masked_images = {field: getattr(self, field) & binary_image for field in fields}
        result = {}
        for field, masked_image in masked_images.items():
            if include_area:
                result[f"{field}_area"] = self.calculate_area(masked_image)
            if include_count:
                result[f"{field}_count"] = self.calculate_count(masked_image)
        return result

    @cached_property
    def dy(self):
        return np.arange(self.h)[:, None] - self.disc_cy

    @cached_property
    def dx(self):
        return np.arange(self.w)[None, :] - self.disc_cx

    @cached_property
    def theta(self):
        return np.arctan2(self.dy, self.dx) / (2 * np.pi)

    @cached_property
    def distance_to_disc(self):
        return self.resolution * np.sqrt(self.dx * self.dx + self.dy * self.dy)

    @cached_property
    def odfd(self):
        return self._odfd

    @cached_property
    def disc_radius_mm(self):
        return self._disc_radius_mm

    @cached_property
    def total(self):
        return np.ones((self.h, self.w), dtype=bool)

    """rings"""

    @cached_property
    def center(self):
        lower = self.disc_radius_mm + 0.2 * self.odfd
        upper = self.disc_radius_mm + 0.4 * self.odfd
        return (lower <= self.distance_to_disc) & (self.distance_to_disc < upper)

    @cached_property
    def inner(self):
        lower = self.disc_radius_mm
        upper = self.disc_radius_mm + 0.2 * self.odfd
        return (lower <= self.distance_to_disc) & (self.distance_to_disc < upper)

    @cached_property
    def outer(self):
        lower = self.disc_radius_mm + 0.4 * self.odfd
        upper = self.disc_radius_mm + 0.6 * self.odfd
        return (lower <= self.distance_to_disc) & (self.distance_to_disc < upper)

    @cached_property
    def grid(self):
        return self.center | self.inner | self.outer

    """quadrants"""

    @cached_property
    def inferior(self):
        return (1 / 8 < self.theta) & (self.theta <= 3 / 8)

    @cached_property
    def left(self):
        return (3 / 8 < self.theta) | (self.theta <= -3 / 8)

    @cached_property
    def superior(self):
        return (-3 / 8 < self.theta) & (self.theta <= -1 / 8)

    @cached_property
    def right(self):
        return (-1 / 8 < self.theta) & (self.theta <= 1 / 8)

    @cached_property
    def nasal(self):
        if self.laterality is None:
            raise RuntimeError("nasal() called in DiscCenteredGrid and laterality not known")
        return self.right if self.laterality == Laterality.R else self.left

    @cached_property
    def temporal(self):
        if self.laterality is None:
            raise RuntimeError(
                "temporal() called in DiscCenteredGrid and laterality not known"
            )
        return self.left if self.laterality == Laterality.R else self.right

    """quadrants grid"""

    @cached_property
    def quadrant_superior(self):
        return self.superior & self.grid

    @cached_property
    def quadrant_inferior(self):
        return self.inferior & self.grid

    @cached_property
    def quadrant_nasal(self):
        return self.nasal & self.grid

    @cached_property
    def quadrant_temporal(self):
        return self.temporal & self.grid

    @cached_property
    def quadrant_left(self):
        return self.left & self.grid

    @cached_property
    def quadrant_right(self):
        return self.right & self.grid

    def plot(self, ax, field: GridFieldEnum | GridField = None):
        """
        Plot the disc-centered grid rings and quadrant lines.
        """
        # Circles at boundary radii (in pixels):
        radii_mm = [
            self.disc_radius_mm,
            self.disc_radius_mm + 0.2 * self.odfd,
            self.disc_radius_mm + 0.4 * self.odfd,
            self.disc_radius_mm + 0.6 * self.odfd,
        ]
        for r_mm in radii_mm:
            r_px = r_mm / self.resolution
            circle = MplCircle(
                (self.disc_cx, self.disc_cy),
                r_px,
                fill=False,
                edgecolor="white",
                linestyle="-",
                linewidth=0.5,
            )
            ax.add_artist(circle)

        # Quadrant separator lines
        for angle in [45, 135, 225, 315]:
            x = [
                self.disc_cx,
                self.disc_cx +  (self.disc_radius_mm + 0.6 * self.odfd) * np.cos(np.radians(angle)) / self.resolution,
            ]
            y = [
                self.disc_cy,
                self.disc_cy + (self.disc_radius_mm + 0.6 * self.odfd) * np.sin(np.radians(angle)) / self.resolution,
            ]
            ax.plot(x, y, color="white", linestyle="-", linewidth=0.5)

        # Highlight specific field if provided
        if field is not None:
            enum = field.enum if isinstance(field, GridField) else field
            mask = getattr(self, enum.value)
            highlighted = np.ma.masked_where(~mask, np.ones_like(mask))
            ax.imshow(highlighted, cmap="gray", alpha=0.2, extent=[0, self.w, self.h, 0])

        ax.set_xlim(0, self.w)
        ax.set_ylim(self.h, 0)
        ax.axis("off")

    # Link enum type to grid
    FieldEnum = DiscCenteredField


