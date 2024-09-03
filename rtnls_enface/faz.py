from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, List, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.spatial import distance

from rtnls_enface.base import Point, TuplePoint
from rtnls_enface.utils.data_loading import open_mask

if TYPE_CHECKING:
    from rtnls_enface.faz_enface import FAZEnface


class FAZ:
    def __init__(self, mask, faz_enface: FAZEnface = None):
        center_of_mass = np.mean(np.argwhere(mask), axis=0).squeeze()
        self.center_of_mass = Point(*center_of_mass)

        # Label each connected component
        labeled_array, num_features = ndimage.label(mask)

        if num_features == 0:
            warnings.warn("Found no connected components for faz mask. ")
        elif num_features >= 2:
            warnings.warn("Found more than one connected component for faz mask. ")
            # Find the size of each connected component
            sizes = ndimage.sum(mask, labeled_array, range(1, num_features + 1))
            # Identify the largest connected component
            max_label = np.argmax(sizes) + 1
            # Create an output image where only the largest component is kept
            mask = np.where(labeled_array == max_label, 1, 0)

        mask = ndimage.binary_fill_holes(mask).astype(np.uint8)

        self.num_ccs = num_features
        self.mask = mask
        self.faz_enface = faz_enface

    @property
    def diameter(self):
        return 2 * self.radius

    def distance_to_center(self, point: TuplePoint):
        return distance.euclidean(self.center_of_mass.tuple, point)

    def contains(self, points: List[TuplePoint]) -> bool:
        return all([bool(self.mask[p[0], p[1]]) for p in points])

    def intersects(self, points: List[TuplePoint]) -> bool:
        return any([bool(self.mask[p[0], p[1]]) for p in points])

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

    def plot(self, ax=None, fig=None):
        if ax is None:
            if self.fundus is not None:
                fig, ax = self.fundus.plot_fundus()
            else:
                fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=300)
                ax.set_axis_off()

        im = self.mask
        cmap = plt.get_cmap("binary")
        cmap.set_bad((0, 0, 0, 0))
        masked = np.ma.masked_where(im == 0, im)
        ax.imshow(masked, cmap=cmap, interpolation="nearest")
        return fig, ax

    @classmethod
    def from_file(cls, fpath: Union[str, Path]):
        im = open_mask(fpath)
        if len(im.shape) == 2:
            mask = (im > 0).astype(np.uint8).squeeze()
        else:
            mask = im[:, :, 0].squeeze()
        return cls(mask)
