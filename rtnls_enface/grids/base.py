from __future__ import annotations

from abc import ABC, abstractmethod
from copy import copy
from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np

from rtnls_enface.base import EnfaceImage, Point, TuplePoint

if TYPE_CHECKING:
    from rtnls_enface.fundus import Fundus


class GridFieldEnum(Enum):
    """Fields of the grid."""

    @staticmethod
    @abstractmethod
    def grid():
        """Return the grid class associated with the field."""
        pass


class Grid(ABC):
    def __init__(self, image: EnfaceImage, resolution, mask=None):
        """
        h: height of the image
        w: width of the image
        fovea_x: x coordinate of the fovea
        fovea_y: y coordinate of the fovea
        resolution: resolution of the image in mm/pix
        laterality: laterality of the eye, 'R' or 'L'
        """
        self.h, self.w = image.resolution
        self.resolution = resolution
        self.mask = mask
        self._full_field_pixel_count_cache: Dict[GridFieldEnum, int] = {}

    def field(self, field: "GridFieldEnum") -> "GridField":
        """Return a GridField bound to this grid for the given field enum."""
        return GridField(self, field)

    def _mask_from(self, field: Union["GridFieldEnum", "GridField"]) -> np.ndarray:
        """Normalize a GridFieldEnum or GridField to its boolean mask array."""
        if isinstance(field, GridField):
            return field.mask
        return getattr(self, field.value)


    @staticmethod
    def _circle_bounds(
        center_x: float, center_y: float, radius: float
    ) -> Tuple[int, int, int, int]:
        """Return an integer canvas that fully contains a circular extent."""
        x0 = int(np.floor(center_x - radius))
        y0 = int(np.floor(center_y - radius))
        x1 = int(np.ceil(center_x + radius)) + 1
        y1 = int(np.ceil(center_y + radius)) + 1
        return x0, y0, x1, y1

    def _full_field_bounds(
        self, field: "GridFieldEnum"
    ) -> Optional[Tuple[int, int, int, int]]:
        """Return a canvas in image coordinates that contains the complete field."""
        return None

    def _shift_canvas_origin(self, x_offset: int, y_offset: int) -> None:
        """Translate geometry into a virtual canvas whose origin is offset."""
        raise NotImplementedError


    @staticmethod
    def _clear_cached_properties(grid: "Grid") -> None:
        for grid_class in type(grid).__mro__:
            for name, descriptor in vars(grid_class).items():
                if isinstance(descriptor, cached_property):
                    grid.__dict__.pop(name, None)

    def full_field_pixel_count(
        self, field: Union["GridFieldEnum", "GridField"]
    ) -> int:
        """Return the rasterized area of a field before image-boundary clipping."""
        field_enum = field.enum if isinstance(field, GridField) else field
        cached = self._full_field_pixel_count_cache.get(field_enum)
        if cached is not None:
            return cached

        bounds = self._full_field_bounds(field_enum)
        if bounds is None:
            count = int(np.count_nonzero(self._mask_from(field_enum)))
        else:
            x0, y0, x1, y1 = bounds
            if x1 <= x0 or y1 <= y0:
                count = 0
            else:
                padded_grid = copy(self)
                self._clear_cached_properties(padded_grid)
                padded_grid.h = y1 - y0
                padded_grid.w = x1 - x0
                padded_grid.mask = None
                padded_grid._shift_canvas_origin(x0, y0)
                count = int(np.count_nonzero(padded_grid._mask_from(field_enum)))

        self._full_field_pixel_count_cache[field_enum] = count
        return count

    def field_visible(self, field: Union["GridFieldEnum", "GridField"]):
        """Return whether the complete geometric field is visible."""
        return self.field_within_bounds(field) == 1.0

    def field_within_bounds(self, field: Union["GridFieldEnum", "GridField"]) -> float:
        """Return the fraction of the geometric field that is within image and mask bounds."""
        f = self._mask_from(field)
        field_pixels = self.full_field_pixel_count(field)
        if field_pixels == 0:
            return 0.0
        within_bounds_pixels = (
            np.count_nonzero(f)
            if self.mask is None
            else np.count_nonzero(f & self.mask)
        )
        return float(within_bounds_pixels) / float(field_pixels)

    def point_in_field(self, point: Point, field: Union["GridFieldEnum", "GridField"]):
        mask = self._mask_from(field)
        p = point.tuple
        return mask[p[0], p[1]]

    def fraction_in_field(self, points: List[TuplePoint], field: Union["GridFieldEnum", "GridField"]):
        mask = self._mask_from(field)
        return np.mean([mask[p[0], p[1]] for p in points])

    def plot(self, ax, field: GridFieldEnum = None):
        raise NotImplementedError()


class GridField:
    """A concrete field bound to a specific Grid instance.

    Provides access to the boolean mask via `.mask`, while exposing the originating
    grid instance through `.grid` and the enum specification via `.enum`.
    """

    def __init__(self, grid: "Grid", field: "GridFieldEnum") -> None:
        grid_field_enum_type = getattr(grid.__class__, "FieldEnum", None)
        if grid_field_enum_type is None or not isinstance(field, grid_field_enum_type):
            raise TypeError(
                "GridFieldEnum type does not match the grid's FieldEnum for this Grid instance"
            )
        self._grid = grid
        self._enum = field

    @property
    def grid(self) -> "Grid":
        return self._grid

    @property
    def enum(self) -> "GridFieldEnum":
        return self._enum

    @property
    def mask(self) -> np.ndarray:
        return getattr(self._grid, self._enum.value)

    def plot(self, ax):
        self._grid.plot(ax, self._enum)

    def full_pixel_count(self) -> int:
        """Return the field area before image-boundary clipping."""
        return self._grid.full_field_pixel_count(self)
