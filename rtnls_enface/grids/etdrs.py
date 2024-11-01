from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.patches import Circle
from skimage import measure

from rtnls_enface.base import Laterality

from .base import Grid, GridField

if TYPE_CHECKING:
    from rtnls_enface.faz_enface import FAZEnface
    from rtnls_enface.fundus import Fundus


class ETDRSField(GridField):
    @staticmethod
    def grid():
        return ETDRSGrid


class MiscField(ETDRSField):
    Total = "total"


class Subfield(ETDRSField):
    CSF = "CSF"
    SIM = "SIM"
    NIM = "NIM"
    TIM = "TIM"
    IIM = "IIM"
    SOM = "SOM"
    NOM = "NOM"
    TOM = "TOM"
    IOM = "IOM"


class Ring(ETDRSField):
    Center = "center"
    Inner = "inner"
    Outer = "outer"


class Quadrant(ETDRSField):
    Superior = "quadrant_superior"
    Inferior = "quadrant_inferior"
    Nasal = "quadrant_nasal"
    Temporal = "quadrant_temporal"
    Right = "quadrant_right"
    Left = "quadrant_left"


class ETDRSGrid(Grid):
    subfields_9 = "CSF", "SIM", "NIM", "TIM", "IIM", "SOM", "NOM", "TOM", "IOM"
    rings_3 = "center", "inner", "outer"
    quadrants = (
        "quadrant_superior",
        "quadrant_nasal",
        "quadrant_inferior",
        "quadrant_temporal",
        "quadrant_right",
        "quadrant_left",
    )
    all_fields = tuple([*subfields_9, *rings_3, *quadrants, "total"])

    def __init__(self, retina: Fundus | FAZEnface, resolution, mask, multiplier=1):
        super().__init__(retina, resolution, mask)
        self.fovea_x = retina.fovea_location.x
        self.fovea_y = retina.fovea_location.y
        self.laterality = retina.laterality
        self.multiplier = multiplier

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
        return np.arange(self.h)[:, None] - self.fovea_y

    @cached_property
    def dx(self):
        return np.arange(self.w)[None, :] - self.fovea_x

    @cached_property
    def theta(self):
        return np.arctan2(self.dy, self.dx) / (2 * np.pi)

    @cached_property
    def distance_to_fovea(self):
        return self.resolution * np.sqrt(self.dx * self.dx + self.dy * self.dy)

    @cached_property
    def total(self):
        return np.ones((self.h, self.w), dtype=bool)

    """rings"""

    @cached_property
    def center(self):
        return self.distance_to_fovea < 0.5 * self.multiplier

    @cached_property
    def inner(self):
        return (self.distance_to_fovea < 1.5 * self.multiplier) & ~self.center

    @cached_property
    def outer(self):
        return (self.distance_to_fovea < 3 * self.multiplier) & ~(
            self.center | self.inner
        )

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
            raise RuntimeError("nasal() called in ETDRSGrid and laterality not known")
        return self.right if self.laterality == Laterality.R else self.left

    @cached_property
    def temporal(self):
        if self.laterality is None:
            raise RuntimeError(
                "temporal() called in ETDRSGrid and laterality not known"
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

    """subfields"""

    @cached_property
    def CSF(self):
        return self.center

    @cached_property
    def SIM(self):
        return self.superior & self.inner

    @cached_property
    def NIM(self):
        return self.nasal & self.inner

    @cached_property
    def TIM(self):
        return self.temporal & self.inner

    @cached_property
    def IIM(self):
        return self.inferior & self.inner

    @cached_property
    def NOM(self):
        return self.nasal & self.outer

    @cached_property
    def TOM(self):
        return self.temporal & self.outer

    @cached_property
    def SOM(self):
        return self.superior & self.outer

    @cached_property
    def IOM(self):
        return self.inferior & self.outer

    def plot(self, ax, field: GridField = None):
        """
        Plot the ETDRS grid regions using white lines to draw circles and lines.

        Parameters:
        - ax: matplotlib axes object to plot on
        - field: Optional ETDRSField to highlight
        """
        # Plot circles
        for radius in [0.5, 1.5, 3]:
            circle = Circle(
                (self.fovea_x, self.fovea_y),
                radius * self.multiplier / self.resolution,
                fill=False,
                edgecolor="white",
                linestyle="-",
                linewidth=1,
            )
            ax.add_artist(circle)

        # Plot lines
        for angle in [45, 135, 225, 315]:
            x = [
                self.fovea_x,
                self.fovea_x
                + 3 * self.multiplier * np.cos(np.radians(angle)) / self.resolution,
            ]
            y = [
                self.fovea_y,
                self.fovea_y
                + 3 * self.multiplier * np.sin(np.radians(angle)) / self.resolution,
            ]
            ax.plot(x, y, color="white", linestyle="-", linewidth=1)

        # Highlight specific field if provided
        if field is not None:
            mask = getattr(self, field.value)
            highlighted = np.ma.masked_where(~mask, np.ones_like(mask))
            ax.imshow(
                highlighted, cmap="gray", alpha=0.2, extent=[0, self.w, self.h, 0]
            )

        ax.set_xlim(0, self.w)
        ax.set_ylim(self.h, 0)
        ax.axis("off")
