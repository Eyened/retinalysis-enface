from __future__ import annotations

import warnings
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, List, Union

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import distance
from scipy.optimize import brentq

from rtnls_enface.base import Circle, Point, TuplePoint
from rtnls_enface.disc_ellipse import EllipseFitError, OpticDiscEllipseFit, fit_disc_ellipse
from rtnls_enface.utils.data_loading import open_mask

if TYPE_CHECKING:
    from rtnls_enface.fundus import Fundus


class InvalidDiscMaskException(Exception):
    pass


class OpticDisc:
    def __init__(self, mask, fundus: Fundus = None, size=1024):
        # Check if mask is square and has the required size
        h, w = mask.shape[:2]

        if h != w:
            raise ValueError("Optic disc mask is not square")

        # Rescale if not the right size
        if h != size:
            # Resize to the required size
            mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)

        # Label each connected component
        mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
        labeled_array, num_features = ndimage.label(mask)

        if num_features == 0:
            raise InvalidDiscMaskException(
                "Found no connected components for disc mask."
            )
        elif num_features >= 2:
            warnings.warn("Found more than one connected component for disc mask. ")
            # Find the size of each connected component
            sizes = ndimage.sum(mask, labeled_array, range(1, num_features + 1))
            # Identify the largest connected component
            max_label = np.argmax(sizes) + 1
            # Create an output image where only the largest component is kept
            mask = np.where(labeled_array == max_label, 1, 0)

        center_of_mass = np.mean(np.argwhere(mask), axis=0).squeeze()
        self.center_of_mass = Point(*center_of_mass)

        self.num_ccs = num_features
        self.mask = mask
        self.fundus = fundus
        self.ellipse = self.fit_ellipse()

    def fit_ellipse(
        self, valid_mask=None, *, boundary_margin_px: float = 5.0,
        roundness_weight: float = 0.02, robust_scale_px: float = 2.0,
    ) -> OpticDiscEllipseFit:
        """Reconstruct a possibly clipped OD ellipse from its visible contour.

        Uses the attached fundus ROI unless ``valid_mask`` is supplied; without
        either, only image edges are known. Pass the CFI mask to censor an internal
        field-of-view boundary. All coordinates/margins refer to ``self.mask``'s
        resolution (normally 1024), not the original acquisition resolution.

        Construction stores one accepted default fit in ``ellipse``. Explicit
        calls return an independently validated fit without replacing it.
        Observed segmentation and its center of mass remain available unchanged.
        """
        if valid_mask is None:
            valid_mask = getattr(self.fundus, "roi_mask", None)
        if valid_mask is None:
            valid_mask = np.ones(self.mask.shape, dtype=bool)
        try:
            fit = fit_disc_ellipse(
                self.mask, valid_mask, boundary_margin_px=boundary_margin_px,
                roundness_weight=roundness_weight, robust_scale_px=robust_scale_px,
            )
        except (EllipseFitError, cv2.error, np.linalg.LinAlgError, FloatingPointError) as exc:
            raise InvalidDiscMaskException(str(exc)) from exc
        if fit.is_poor_fit:
            raise InvalidDiscMaskException("Optic-disc mask does not fit an ellipse")
        return fit

    @property
    def center(self) -> Point:
        """Center of the reconstructed disc, potentially outside the image."""
        return self.ellipse.center

    def _disc_axis_unit_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """Return unit vectors for the temporal axis and its superior direction."""
        if self.fundus is None or self.fundus.fovea_location is None:
            raise ValueError("Fundus with fovea location must be set for disc side points")

        center = self.center
        fovea = self.fundus.fovea_location
        axis = np.array([fovea.y - center.y, fovea.x - center.x], dtype=float)
        norm = np.linalg.norm(axis)
        if norm == 0:
            raise ValueError("Fovea location must differ from disc center")

        temporal_unit = axis / norm

        # Image y grows downward: choose the upper perpendicular in either eye.
        superior_unit = np.array([-temporal_unit[1], temporal_unit[0]], dtype=float)
        if superior_unit[0] > 0:
            superior_unit = -superior_unit
        return temporal_unit, superior_unit

    def _project_disc_side_point(self, direction: np.ndarray) -> Point:
        """Project a unit direction from the disc center onto the disc boundary."""
        center = self.center
        theta = np.deg2rad(self.ellipse.angle_degrees)
        dy, dx = direction
        u = dx * np.cos(theta) + dy * np.sin(theta)
        v = -dx * np.sin(theta) + dy * np.cos(theta)
        a, b = self.ellipse.semi_axes
        offset = direction / np.hypot(u / a, v / b)
        return Point(center.y + offset[0], center.x + offset[1])

    @cached_property
    def circle(self) -> Circle:
        """Area-equivalent circle centered on the reconstructed ellipse."""
        return Circle(center=self.center, r=self.ellipse.equivalent_radius)

    @cached_property
    def temporal_point(self) -> Point:
        """Tilt-aware temporal disc boundary point along the OD-fovea axis."""
        temporal_unit, _ = self._disc_axis_unit_vectors()
        return self._project_disc_side_point(temporal_unit)

    @cached_property
    def nasal_point(self) -> Point:
        """Tilt-aware nasal disc boundary point opposite the OD-fovea axis."""
        temporal_unit, _ = self._disc_axis_unit_vectors()
        return self._project_disc_side_point(-temporal_unit)

    @cached_property
    def superior_point(self) -> Point:
        """Tilt-aware superior disc boundary point perpendicular to the OD-fovea axis."""
        _, superior_unit = self._disc_axis_unit_vectors()
        return self._project_disc_side_point(superior_unit)

    @cached_property
    def inferior_point(self) -> Point:
        """Tilt-aware inferior disc boundary point perpendicular to the OD-fovea axis."""
        _, superior_unit = self._disc_axis_unit_vectors()
        return self._project_disc_side_point(-superior_unit)

    @cached_property
    def bounding_box(self):
        """
        Given a binary mask with a single blob, returns the bounding box as (x0, x1, y0, y1).

        Parameters:
        - mask: 2D numpy array, binary mask where the blob is represented by True (or 1)

        Returns:
        - (x0, x1, y0, y1): coordinates of the bounding box
        """
        rows = np.any(self.mask, axis=1)
        cols = np.any(self.mask, axis=0)

        y0, y1 = np.where(rows)[0][[0, -1]]
        x0, x1 = np.where(cols)[0][[0, -1]]

        return x0, x1, y0, y1

    @cached_property
    def square_bb(self):
        x0, x1, y0, y1 = self.bounding_box
        width = x1 - x0
        height = y1 - y0
        side = max(width, height)

        x_center = (x0 + x1) / 2
        y_center = (y0 + y1) / 2

        x0_new = x_center - side / 2
        x1_new = x_center + side / 2
        y0_new = y_center - side / 2
        y1_new = y_center + side / 2

        return int(x0_new), int(x1_new), int(y0_new), int(y1_new)

    def get_padded_square_bb(self, padding_percent=10):
        """
        Returns a square bounding box expanded by a percentage of its side.

        Parameters:
        - padding_percent: percentage of the side length to pad on each side

        Returns:
        - (x0, x1, y0, y1): coordinates of the padded square bounding box
        """
        x0, x1, y0, y1 = self.square_bb
        side = x1 - x0  # Square side length

        # Calculate padding amount
        padding = side * (padding_percent / 100)

        # Add padding to all sides
        x0_padded = max(0, int(x0 - padding))
        x1_padded = int(x1 + padding)
        y0_padded = max(0, int(y0 - padding))
        y1_padded = int(y1 + padding)

        return x0_padded, x1_padded, y0_padded, y1_padded

    def distance_to_center(self, point: TuplePoint):
        return distance.euclidean(self.center.tuple, point)

    def contains(self, points: List[TuplePoint]) -> bool:
        return all([bool(self.mask[p[0], p[1]]) for p in points])

    def intersects(self, points: List[TuplePoint]) -> bool:
        return any([bool(self.mask[p[0], p[1]]) for p in points])

    @cached_property
    def boundary_pixels(self) -> np.ndarray:
        """Contour pixels of the disc mask used for distance queries."""
        contours, _ = cv2.findContours(
            self.mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        if len(contours) == 0:
            return np.empty((0, 2), dtype=int)
        contour = np.concatenate(contours, axis=0).reshape(-1, 2)
        return contour[:, ::-1]

    def distance_to_bounds(self):
        """Return the minimum distance from the disc contour to the ROI boundary."""
        if self.fundus is None or self.fundus.roi_mask is None:
            raise ValueError("ROI mask must be set for distance to bounds")

        boundary_pixels = self.boundary_pixels
        if boundary_pixels.size == 0:
            raise ValueError("Disc boundary could not be computed")

        distances = self.fundus.mask_distance_transform[
            boundary_pixels[:, 0], boundary_pixels[:, 1]
        ]
        return float(np.min(distances))

    def distance_to_visible_bounds(self) -> float:
        """Return the minimum distance from the disc contour to ROI or image bounds."""
        x0, x1, y0, y1 = self.bounding_box
        h, w = self.mask.shape[:2]
        image_edge_distance = min(x0, y0, w - 1 - x1, h - 1 - y1)

        if self.fundus is None or self.fundus.roi_mask is None:
            return float(image_edge_distance)

        return float(min(image_edge_distance, self.distance_to_bounds()))

    def closest_point(self, point: Point) -> Point:
        """Nearest reconstructed boundary point, including for interior queries."""
        theta = np.deg2rad(self.ellipse.angle_degrees)
        c, s = np.cos(theta), np.sin(theta)
        dx, dy = point.x - self.center.x, point.y - self.center.y
        u, v = dx * c + dy * s, -dx * s + dy * c
        a, b = self.ellipse.semi_axes
        x, y = abs(u), abs(v)
        # In the first quadrant the Lagrange multiplier has a unique root.
        # Handle the major-axis singularity explicitly (also covers the center).
        if y < 1e-12:
            if a == b or a * x >= a * a - b * b:
                qx, qy = a, 0.0
            else:
                qx = a * a * x / (a * a - b * b)
                qy = b * np.sqrt(max(0.0, 1 - (qx / a) ** 2))
        else:
            # z = lambda + b² keeps the inside root away from cancellation.
            def equation(z):
                return (a * x / (z + a*a - b*b))**2 + (b * y / z)**2 - 1
            z = brentq(equation, max(np.finfo(float).tiny, b*y/2),
                       b*b + np.hypot(a*x, b*y), xtol=1e-12)
            qx, qy = a*a*x / (z + a*a - b*b), b*b*y / z
        qx, qy = np.copysign(qx, u), np.copysign(qy, v)
        return Point(self.center.y + qx*s + qy*c,
                     self.center.x + qx*c - qy*s)

    @classmethod
    def from_file(cls, fpath: Union[str, Path]):
        im = open_mask(fpath)
        if len(im.shape) == 2:
            disc_mask = (im > 0).astype(np.uint8).squeeze()
        else:
            disc_mask = im[:, :, 0].squeeze()
        return cls(disc_mask)
