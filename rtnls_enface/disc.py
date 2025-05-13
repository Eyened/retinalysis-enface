from __future__ import annotations

import warnings
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, List, Union

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import distance

from rtnls_enface.base import Circle, Point, TuplePoint
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

    @cached_property
    def circle(self):
        if self.mask is None:
            raise ValueError(
                "disc segmentation not provided. Cannot calculate disc parameters"
            )
        contours, _ = cv2.findContours(
            self.mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        assert len(contours) == 1, f"Found {len(contours)} in disc, expected 1"

        # calculate moments for each contour
        disc = contours[0].squeeze()
        M = cv2.moments(disc)

        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        radius = np.mean(np.sqrt(np.sum((disc - np.array([cX, cY])) ** 2, axis=1)))
        return Circle(center=Point(cY, cX), r=radius)

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
        return distance.euclidean(self.center_of_mass.tuple, point)

    def contains(self, points: List[TuplePoint]) -> bool:
        return all([bool(self.mask[p[0], p[1]]) for p in points])

    def intersects(self, points: List[TuplePoint]) -> bool:
        return any([bool(self.mask[p[0], p[1]]) for p in points])

    def distance_to_bounds(self):
        # Unpack circle center for clarity
        x, y = self.fundus.bounds.cx, self.fundus.bounds.cy
        circle_radius = self.fundus.bounds.radius

        # Generate a grid of coordinates
        xx, yy = np.meshgrid(
            np.arange(self.mask.shape[1]), np.arange(self.mask.shape[0])
        )

        # Calculate distances from each point to the circle's center
        dist_to_center = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)

        # Calculate distances to the circle's edge
        dist_to_edge = np.abs(dist_to_center - circle_radius)

        # Mask the distance map with the binary image
        masked_dist = np.where(self.mask == 1, dist_to_edge, np.inf)

        # Find the minimum distance and its location
        min_dist_idx = np.unravel_index(np.argmin(masked_dist), masked_dist.shape)

        # Return the location of the closest "on" point and the distance
        return min_dist_idx, masked_dist[min_dist_idx]

    def distance_to_bounds_2(self):
        if self.fundus.bounds is None:
            raise ValueError("Bounds must be set for distance to bounds")
        cx, cy = self.fundus.bounds.cx, self.fundus.bounds.cy
        r = self.fundus.bounds.radius
        # Convert the image to grayscale if it is not already
        if len(self.mask.shape) == 3:
            self.mask = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)

        # Find contours in the image
        contours, _ = cv2.findContours(
            self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Calculate minimum distances
        min_distances = []
        for contour in contours:
            for point in contour:
                # Extract point coordinates
                x, y = point.ravel()
                # Calculate distance from point to the circle center
                distance_to_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                # Calculate absolute difference between distance to center and radius
                distance_to_circle = abs(distance_to_center - r)
                min_distances.append(distance_to_circle)

        return np.min(min_distances)

    def closest_point(self, point: Point) -> Point:
        # Identify the blob points
        blob_points = np.argwhere(self.mask > 0)

        # Calculate distances from each blob point to the target point
        distances = np.sqrt(((blob_points - point.tuple) ** 2).sum(axis=1))

        # Find the index of the closest point
        closest_point_index = np.argmin(distances)

        # Return the closest blob point
        p = blob_points[closest_point_index]
        return Point(p[0], p[1])

    @classmethod
    def from_file(cls, fpath: Union[str, Path]):
        im = open_mask(fpath)
        if len(im.shape) == 2:
            disc_mask = (im > 0).astype(np.uint8).squeeze()
        else:
            disc_mask = im[:, :, 0].squeeze()
        return cls(disc_mask)
