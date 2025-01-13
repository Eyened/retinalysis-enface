from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional, Set, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance

if TYPE_CHECKING:
    from vascx.fundus.retina import Retina


class EnfaceImage(ABC):
    def __init__(
        self,
        image: np.array = None,
        id: Any = None,
    ):
        self.image = image
        self.id = id
        if image is not None:
            self.resolution = tuple(image.shape[:2])

    def plot_image(self, ax=None, fig=None):
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=300)
            ax.set_axis_off()

        if self.image is None:
            return

        ax.imshow(self.image)

        return ax

    @classmethod
    @abstractmethod
    def from_file(cls, *args, **kwargs):
        pass


class Layer:
    pass


class FundusQCValidator:
    pass


class FundusQCGrader:
    pass


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

    @property
    def numpy(self):
        return np.array([self.x, self.y], dtype=float)

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
        v1 = self.p2.numpy - self.p1.numpy
        v2 = line.p2.numpy - line.p1.numpy

        unit_v1 = v1 / np.linalg.norm(v1)
        unit_v2 = v2 / np.linalg.norm(v2)

        # Calculate the angle in radians
        angle_radians = np.arccos(np.dot(unit_v1, unit_v2))

        # Convert the angle to degrees
        angle_degrees = np.degrees(angle_radians)

        # Since lines are undirected, the smallest angle can be the complement to 180 if it's larger than 90 degrees
        return angle_degrees

    def counterclockwise_angle_to(self, line: Line):
        """
        Calculate the counterclockwise angle from the first line to the second line.

        Returns:
            angle (float): The counterclockwise angle between the two lines in degrees.
        """

        # Calculate slopes
        x1, y1 = self.p1.tuple_xy
        x2, y2 = self.p2.tuple_xy
        x3, y3 = line.p1.tuple_xy
        x4, y4 = line.p2.tuple_xy

        # Calculate the direction vectors of the two lines
        dx1, dy1 = x2 - x1, y2 - y1
        dx2, dy2 = x4 - x3, y4 - y3

        # Calculate the angle in radians using arctan2
        angle1 = np.arctan2(dy1, dx1)
        angle2 = np.arctan2(dy2, dx2)

        # Calculate the counterclockwise angle
        angle_rad = angle2 - angle1

        # Convert the angle to degrees
        angle_deg = np.degrees(angle_rad)

        # Adjust the angle to ensure it's between 0 and 360 degrees
        if angle_deg < 0:
            angle_deg += 360

        return angle_deg

    def point_at(self, fraction: float):
        p1, p2 = self.p1, self.p2
        x = p1.x + (p2.x - p1.x) * fraction
        y = p1.y + (p2.y - p1.y) * fraction
        return Point(y, x)

    @property
    def length(self):
        return np.sqrt((self.p1.x - self.p2.x) ** 2 + (self.p1.y - self.p2.y) ** 2)

    def plot(self, ax, **kwargs):
        ax.plot([self.p1.x, self.p2.x], [self.p1.y, self.p2.y], **kwargs)


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


class Laterality(Enum):
    R = 0
    L = 1
