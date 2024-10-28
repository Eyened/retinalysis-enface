from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, List

import numpy as np

from rtnls_enface.base import EnfaceImage, Point, TuplePoint

if TYPE_CHECKING:
    from rtnls_enface.fundus import Fundus


class GridField(Enum):
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

    def field(self, field: GridField) -> np.ndarray:
        return getattr(self, field.value)

    def field_visible(self, field: GridField):
        """Return True if the field is completely visible/contained in the fundus image."""
        f = self.field(field)
        return np.all(f <= self.mask)

    def point_in_field(self, point: Point, field: GridField):
        field = self.field(field)
        p = point.tuple
        return field[p[0], p[1]]

    def fraction_in_field(self, points: List[TuplePoint], field: GridField):
        field = self.field(field)
        return np.mean([field[p[0], p[1]] for p in points])

    def plot(self, ax, field: GridField = None):
        raise NotImplementedError()
