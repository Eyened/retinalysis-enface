from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Dict, Optional, Set, Tuple, Union

import numpy as np
from scipy.spatial import distance

if TYPE_CHECKING:
    from vascx.retina import Retina


TuplePoint = Union[Tuple[int, int], Tuple[float, float]]


class LayerType(Enum):
    ARTERIES = 1
    VEINS = 2
    VESSELS = 3


@dataclass
class Point:
    y: int or float
    x: int or float
    retina: Optional[Retina] = None

    def is_in_retina(self):
        return (
            0 <= self.x < self.retina.resolution[0]
            and 0 <= self.y < self.retina.resolution[1]
        )

    @property
    def tuple(self):
        return (self.y, self.x)

    @property
    def tuple_xy(self):
        return (self.x, self.y)

    def distance_to(self, point: Point):
        return distance.euclidean(self.tuple, point.tuple)

    def __eq__(self, other):
        # Ensure that 'other' is of the same type and has the same value
        return isinstance(other, Point) and self.x == other.x and self.y == other.y

    def __hash__(self):
        # Return the hash value of the attribute that defines equality
        return hash((self.x, self.y))


@dataclass
class Line:
    p1: Point
    p2: Point

    def orientation(self):
        x, y = self.p2.x - self.p1.x, self.p2.y - self.p1.y
        angle = math.atan2(y, x)
        return math.degrees(angle)

    def angle_to(self, line: Line):
        # Convert points to vectors
        v1 = np.array(self.p2.tuple_xy) - np.array(self.p1.tuple_xy)
        v2 = np.array(line.p2.tuple_xy) - np.array(line.p1.tuple_xy)

        # Calculate the dot product of v1 and v2
        dot_product = np.dot(v1, v2)

        # Calculate the magnitude of v1 and v2
        magnitude_v1 = np.linalg.norm(v1)
        magnitude_v2 = np.linalg.norm(v2)

        # Calculate the cosine of the angle between v1 and v2
        cos_angle = dot_product / (magnitude_v1 * magnitude_v2)

        # Ensure the cosine value is within the valid range
        cos_angle = np.clip(cos_angle, -1, 1)

        # Calculate the angle in radians
        angle_radians = np.arccos(cos_angle)

        # Convert the angle to degrees
        angle_degrees = np.degrees(angle_radians)

        # Since lines are undirected, the smallest angle can be the complement to 180 if it's larger than 90 degrees
        return angle_degrees

    def point_at(self, fraction: float):
        p1, p2 = self.p1, self.p2
        x = p1.x + (p2.x - p1.x) * fraction
        y = p1.y + (p2.y - p1.y) * fraction
        return Point(y, x)

    @property
    def length(self):
        return np.sqrt((self.p1.x - self.p2.x) ** 2 + (self.p1.y - self.p2.y) ** 2)


@dataclass
class Circle:
    center: Point
    r: float
    retina: Optional[Retina] = None

    def fraction_in_retina(self):
        """Returns the fraction of the circle contained in the retina (0-1)
            Parameters:
        - cx (float): The x-coordinate of the circle's center.
        - cy (float): The y-coordinate of the circle's center.
        - r (float): The radius of the circle.
        - width (int): The width of the image.
        - height (int): The height of the image.

        Returns:
        - float: An estimated portion of the circle (0-1) that is contained within the image.
        """
        # Check if the circle is fully inside the image

        width, height = self.retina.resolution
        cx, cy = self.center.tuple

        if (
            (0 <= cx - self.r)
            and (cx + self.r <= width)
            and (0 <= cy - self.r)
            and (cy + self.r <= height)
        ):
            return 1.0  # Circle is fully inside
        # Check if the circle is fully outside the image
        elif (
            (cx + self.r < 0)
            or (cx - self.r > width)
            or (cy + self.r < 0)
            or (cy - self.r > height)
        ):
            return 0.0  # Circle is fully outside
        # For partial overlap, a complex calculation is needed. Here, we return a simple estimate.
        else:
            # Simple estimation for partial overlap
            # This part can be significantly improved with a more accurate calculation
            overlap_width = min(cx + self.r, width) - max(cx - self.r, 0)
            overlap_height = min(cy + self.r, height) - max(cy - self.r, 0)
            overlap_area = overlap_width * overlap_height
            circle_area = np.pi * self.r * self.r
            return min(1.0, overlap_area / circle_area)

    def contains(self, point: Point):
        distance = (
            (point.x - self.center.x) ** 2 + (point.y - self.center.y) ** 2
        ) ** 0.5
        return distance <= self.r


@dataclass
class Ellipse:
    center: Point
    width: float
    height: float
    angle: float
    retina: Optional[Retina] = None


BinaryImage = np.ndarray
PixelGraph = Dict[TuplePoint, Set[TuplePoint]]
