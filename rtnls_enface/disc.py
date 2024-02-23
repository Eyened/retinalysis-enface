from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from rtnls_enface.utils.data_loading import open_mask
from scipy import ndimage
from scipy.spatial import distance

from .types import Circle, Point, TuplePoint

if TYPE_CHECKING:
    from rtnls_enface.fundus import Fundus


class OpticDisc:
    def __init__(self, mask, fundus: Fundus = None):
        center_of_mass = np.mean(np.argwhere(mask), axis=0).squeeze()
        self.center_of_mass = Point(*center_of_mass)

        self._circle: Circle = None

        # Label each connected component
        labeled_array, num_features = ndimage.label(mask)

        if num_features == 0:
            raise RuntimeError("Found no connected components for disc mask. ")
        elif num_features >= 2:
            warnings.warn("Found more than one connected component for disc mask. ")
            # Find the size of each connected component
            sizes = ndimage.sum(mask, labeled_array, range(1, num_features + 1))
            # Identify the largest connected component
            max_label = np.argmax(sizes) + 1
            # Create an output image where only the largest component is kept
            mask = np.where(labeled_array == max_label, 1, 0)

        mask = ndimage.binary_fill_holes(mask).astype(np.uint8)

        self.num_ccs = num_features
        self.mask = mask
        self.fundus = fundus

    @property
    def circle(self):
        if self._circle is None:
            self.calc_disc()
        return self._circle

    @property
    def diameter(self):
        return 2 * self.radius

    def distance_to_center(self, point: TuplePoint):
        return distance.euclidean(self.center_of_mass.tuple, point)

    def contains(self, points: List[TuplePoint]) -> bool:
        return all([bool(self.mask[p[0], p[1]]) for p in points])

    def intersects(self, points: List[TuplePoint]) -> bool:
        return any([bool(self.mask[p[0], p[1]]) for p in points])

    def calc_disc(self):
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
        self._circle = Circle(center=Point(cY, cX), r=radius)

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

    def plot(self, ax=None, fig=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=300)
            ax.set_axis_off()

        ax.imshow(self.mask)

    @classmethod
    def from_file(cls, fpath: str | Path):
        im = open_mask(fpath)
        if len(im.shape) == 2:
            disc_mask = (im > 0).astype(np.uint8).squeeze()
        else:
            disc_mask = im[:, :, 0].squeeze()
        return cls(disc_mask)
