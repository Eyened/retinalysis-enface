from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.patches import Circle as MplCircle
from skimage import measure

from rtnls_enface.base import Laterality, Line

from .base import Grid, GridField, GridFieldEnum

if TYPE_CHECKING:
    from rtnls_enface.fundus import Fundus


class DiscCenteredField(GridFieldEnum):
    @staticmethod
    def grid():
        return DiscCenteredGrid

class DiscCenteredHemifield(DiscCenteredField):
    Superior = "superior", "superior region"
    Inferior = "inferior", "inferior region"
    Temporal = "temporal", "temporal region"
    Nasal = "nasal", "nasal region"


class DiscCenteredRing(DiscCenteredField):
    FullGrid = "grid", "full region"
    Center = "center", "middle ring"
    Inner = "inner", "inner ring"
    Outer = "outer", "outer ring"


class DiscCenteredQuadrant(DiscCenteredField):
    Superior = "quadrant_superior", "superior quadrant"
    Inferior = "quadrant_inferior", "inferior quadrant"
    Nasal = "quadrant_nasal", "nasal quadrant"
    Temporal = "quadrant_temporal", "temporal quadrant"
    Right = "quadrant_right", "right quadrant"
    Left = "quadrant_left", "left quadrant"


class DiscCenteredGrid(Grid):
    def __init__(
        self,
        retina: Fundus,
        resolution,
        mask,
        multiplier=1,
        band_crop_fraction: float = 0.0,
    ):
        super().__init__(retina, resolution, mask)
        self.retina = retina
        self.band_crop_fraction = self._validate_band_crop_fraction(
            band_crop_fraction
        )
        # Anchor at disc center
        self.disc_cx = retina.disc.circle.center.x
        self.disc_cy = retina.disc.circle.center.y
        self.laterality = retina.laterality
        self.multiplier = float(multiplier)
        # Physical distances (mm)
        self._odfd = self.resolution * retina.disc_fovea_distance
        self._disc_radius_mm = self.resolution * retina.disc.circle.r
        od_to_fovea = Line(retina.disc.center, retina.fovea_location)
        self._angle_deg = float(od_to_fovea.orientation())

    @staticmethod
    def _validate_band_crop_fraction(value: float) -> float:
        """Validate the fraction cropped from each side of the grid."""
        value = float(value)
        if not 0.0 <= value < 0.5:
            raise ValueError("band_crop_fraction must be in the range [0.0, 0.5)")
        return value

    def _full_field_bounds(self, field: GridFieldEnum):
        radius = (
            self.disc_radius_mm + 0.6 * self.multiplier * self.odfd
        ) / self.resolution
        return self._circle_bounds(self.disc_cx, self.disc_cy, radius)

    def _shift_canvas_origin(self, x_offset: int, y_offset: int) -> None:
        self.disc_cx -= x_offset
        self.disc_cy -= y_offset

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
    def _theta(self) -> float:
        theta = np.radians(self._angle_deg)
        if np.cos(theta) < 0.0:
            theta += np.pi
        return float(theta)

    @cached_property
    def _perp_unit(self) -> tuple[float, float]:
        theta = self._theta
        return -float(np.sin(theta)), float(np.cos(theta))

    @cached_property
    def _perp_coord(self) -> np.ndarray:
        px, py = self._perp_unit
        return self.dx * px + self.dy * py

    @cached_property
    def _axis_coord(self) -> np.ndarray:
        # Signed coordinate along the OD-fovea axis, toward the fovea.
        theta = np.radians(self._angle_deg)
        return self.dx * np.cos(theta) + self.dy * np.sin(theta)

    @cached_property
    def _band_crop_mask(self) -> np.ndarray:
        """Mask that removes bands parallel to the OD-fovea axis."""
        if self.band_crop_fraction == 0.0:
            return np.ones((self.h, self.w), dtype=bool)

        # `_perp_coord` is expressed in pixels, whereas the ring geometry
        # above is expressed in millimetres. Convert the outer radius to
        # pixels before applying the CircleGrid-style crop.
        outer_radius_px = (
            self.disc_radius_mm + 0.6 * self.multiplier * self.odfd
        ) / self.resolution
        allowed_half_width = outer_radius_px * (
            1.0 - 2.0 * self.band_crop_fraction
        )
        return np.abs(self._perp_coord) <= allowed_half_width

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
        lower = self.disc_radius_mm + 0.2 * self.multiplier * self.odfd
        upper = self.disc_radius_mm + 0.4 * self.multiplier * self.odfd
        return (
            (lower <= self.distance_to_disc)
            & (self.distance_to_disc < upper)
            & self._band_crop_mask
        )

    @cached_property
    def inner(self):
        lower = self.disc_radius_mm
        upper = self.disc_radius_mm + 0.2 * self.multiplier * self.odfd
        return (
            (lower <= self.distance_to_disc)
            & (self.distance_to_disc < upper)
            & self._band_crop_mask
        )

    @cached_property
    def outer(self):
        lower = self.disc_radius_mm + 0.4 * self.multiplier * self.odfd
        upper = self.disc_radius_mm + 0.6 * self.multiplier * self.odfd
        return (
            (lower <= self.distance_to_disc)
            & (self.distance_to_disc < upper)
            & self._band_crop_mask
        )

    @cached_property
    def grid(self):
        return self.center | self.inner | self.outer

    """hemifields"""

    @cached_property
    def superior(self):
        return self.grid & (self._perp_coord < 0)

    @cached_property
    def inferior(self):
        return self.grid & ~(self._perp_coord < 0)

    @cached_property
    def temporal(self):
        return self.grid & (self._axis_coord > 0)

    @cached_property
    def nasal(self):
        return self.grid & ~(self._axis_coord > 0)

    """quadrants"""

    @cached_property
    def right(self):
        return (-1 / 8 < self.theta) & (self.theta <= 1 / 8)

    @cached_property
    def left(self):
        return (3 / 8 < self.theta) | (self.theta <= -3 / 8)

    @cached_property
    def quadrant_superior(self):
        return ((-3 / 8 < self.theta) & (self.theta <= -1 / 8)) & self.grid

    @cached_property
    def quadrant_inferior(self):
        return ((1 / 8 < self.theta) & (self.theta <= 3 / 8)) & self.grid

    @cached_property
    def quadrant_nasal(self):
        if self.laterality is None:
            raise RuntimeError(
                "nasal() called in DiscCenteredGrid and laterality not known"
            )
        return self.quadrant_right if self.laterality == Laterality.R else self.quadrant_left

    @cached_property
    def quadrant_temporal(self):
        if self.laterality is None:
            raise RuntimeError(
                "temporal() called in DiscCenteredGrid and laterality not known"
            )
        return self.quadrant_left if self.laterality == Laterality.R else self.quadrant_right

    @cached_property
    def quadrant_left(self):
        return self.left & self.grid

    @cached_property
    def quadrant_right(self):
        return self.right & self.grid

    def plot(self, ax, field: GridFieldEnum | GridField = None):
        """
        Plot the disc-centered grid rings and the requested field split.
        """
        # Circles at boundary radii (in pixels):
        if self.band_crop_fraction == 0.0:
            radii_mm = [
                self.disc_radius_mm,
                self.disc_radius_mm + 0.2 * self.multiplier * self.odfd,
                self.disc_radius_mm + 0.4 * self.multiplier * self.odfd,
                self.disc_radius_mm + 0.6 * self.multiplier * self.odfd,
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
        else:
            for ring_mask in (self.inner, self.center, self.outer):
                ax.contour(
                    ring_mask.astype(float),
                    levels=[0.5],
                    colors="white",
                    linewidths=0.5,
                )

        radius_px = (
            self.disc_radius_mm + 0.6 * self.multiplier * self.odfd
        ) / self.resolution
        field_enum = field.enum if isinstance(field, GridField) else field
        if isinstance(field_enum, DiscCenteredQuadrant):
            # Legacy quadrant fields are divided by two crossing diameters.
            for angle in [45, 135, 225, 315]:
                angle_rad = np.radians(angle)
                x = self.disc_cx + radius_px * np.cos(angle_rad)
                y = self.disc_cy + radius_px * np.sin(angle_rad)
                if self.band_crop_fraction != 0.0:
                    distances = np.linspace(
                        0.0, radius_px, max(2, int(radius_px) + 1)
                    )
                    xs = self.disc_cx + distances * np.cos(angle_rad)
                    ys = self.disc_cy + distances * np.sin(angle_rad)
                    xi = np.rint(xs).astype(int)
                    yi = np.rint(ys).astype(int)
                    in_image = (0 <= xi) & (xi < self.w) & (0 <= yi) & (yi < self.h)
                    in_crop = np.zeros_like(in_image, dtype=bool)
                    in_crop[in_image] = self._band_crop_mask[yi[in_image], xi[in_image]]
                    visible = in_image & in_crop
                    if np.any(visible):
                        last_visible = np.flatnonzero(visible)[-1]
                        x = xs[last_visible]
                        y = ys[last_visible]
                ax.plot(
                    [self.disc_cx, x],
                    [self.disc_cy, y],
                    color="white",
                    linestyle="-",
                    linewidth=0.5,
                )
        else:
            # Hemifields are split by the OD-fovea diameter.
            angle_rad = np.radians(self._angle_deg)
            c, s = np.cos(angle_rad), np.sin(angle_rad)
            ax.plot(
                [self.disc_cx - radius_px * c, self.disc_cx + radius_px * c],
                [self.disc_cy - radius_px * s, self.disc_cy + radius_px * s],
                color="white",
                linestyle="-",
                linewidth=0.5,
            )

        # Highlight specific field if provided
        if field is not None:
            mask = self._mask_from(field)
            highlighted = np.ma.masked_where(~mask, np.ones_like(mask))
            ax.imshow(
                highlighted, cmap="gray", alpha=0.2, extent=[0, self.w, self.h, 0]
            )

        ax.set_xlim(0, self.w)
        ax.set_ylim(self.h, 0)
        ax.axis("off")

    # Link enum type to grid
    FieldEnum = DiscCenteredField
