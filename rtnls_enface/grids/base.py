from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, List, Union

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

    def field(self, field: "GridFieldEnum") -> "GridField":
        """Return a GridField bound to this grid for the given field enum."""
        return GridField(self, field)

    def _mask_from(self, field: Union["GridFieldEnum", "GridField"]) -> np.ndarray:
        """Normalize a GridFieldEnum or GridField to its boolean mask array."""
        if isinstance(field, GridField):
            return field.mask
        return getattr(self, field.value)

    def field_visible(self, field: Union["GridFieldEnum", "GridField"]):
        """Return True if the field is completely visible/contained in the fundus image."""
        f = self._mask_from(field)
        if self.mask is None:
            return True
        return np.all(f <= self.mask)

    def field_within_bounds(self, field: Union["GridFieldEnum", "GridField"]) -> float:
        """Return the fraction of the field that is within the mask bounds."""
        f = self._mask_from(field)
        if self.mask is None:
            return 1.0
        # Count pixels in field that are also within mask
        field_pixels = np.sum(f)
        if field_pixels == 0:
            return 0.0
        within_bounds_pixels = np.sum(f & self.mask)
        return within_bounds_pixels / field_pixels

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
