from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Tuple, Union

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from rtnls_enface.base import EnfaceImage, Layer, Point
from rtnls_enface.disc import OpticDisc
from rtnls_enface.utils.data_loading import open_binary_mask, open_mask
from rtnls_enface.utils.image import match_resolution

if TYPE_CHECKING:
    from rtnls_fundusprep.mask_extraction import Bounds


class Fundus(EnfaceImage):
    def __init__(
        self,
        disc_path_or_mask: Union[str, Path, np.array] = None,
        fundus_path_or_mask: Union[str, Path, np.array] = None,
        layers: Dict[str, Layer] = None,
        fovea_location: Tuple[float, float] = None,
        bounds: Bounds = None,
        scaling_factor=1.0,
        resolution=(1024, 1024),
        id: Any = None,
    ):
        image = self._load_fundus_image(fundus_path_or_mask)
        super().__init__(image, id)

        self.resolution = resolution
        self.load_disc(disc_path_or_mask)

        self.layers = layers if layers is not None else dict()
        for layer in self.layers.values():
            layer.retina = self

        self.fovea_location = (
            Point(fovea_location[1], fovea_location[0])
            if fovea_location is not None
            else None
        )
        self.bounds = bounds
        self.scaling_factor = scaling_factor

    def zone(self, zone_name):
        layers = {
            key: Layer(val.extract_zone(zone_name), self)
            for key, val in self.layers.items()
        }
        return Fundus(layers, self.disc_mask, self.fundus_image)

    def _load_fundus_image(self, path_or_array: Union[None, str, np.array]):
        if path_or_array is None:
            return None
        else:
            if isinstance(path_or_array, np.ndarray):
                return path_or_array
            else:
                return open_mask(path_or_array)

    def load_disc(self, path_or_array: Union[None, str, np.array]):
        if path_or_array is None:
            self.disc_path = None
            self.disc = None
        else:
            if isinstance(path_or_array, np.ndarray):
                disc_mask = path_or_array
                self.disc_path = None
            else:
                disc_mask = open_binary_mask(path_or_array)
                self.disc_path = path_or_array

            disc_mask = match_resolution(
                disc_mask, self.resolution, interpolation=cv2.INTER_NEAREST
            )
            self.disc = OpticDisc(disc_mask, fundus=self)

    @cached_property
    def disc_fovea_distance(self):
        if self.disc is None or self.fovea_location is None:
            raise ValueError("Disc or fovea location not set")
        return self.disc.center_of_mass.distance_to(self.fovea_location)

    @cached_property
    def mask(self) -> np.ndarray:
        if self.bounds is None:
            raise ValueError("Bounds not set")
        return self.bounds.make_binary_mask()

    @cached_property
    def laplacian(self) -> np.ndarray:
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        # Apply the mask to the image
        laplacian[self.mask == 0] = np.nan
        return laplacian

    def set_bounds(self, bounds: Bounds):
        self.bounds = bounds

    def etdrs_point(self) -> Point:
        if self.disc is None or self.fovea_location is None:
            return None
        return self.disc.closest_point(self.fovea_location)

    def plot_fundus(self, ax=None, fig=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=300)
            ax.set_axis_off()

        if self.fundus_image is None:
            return

        ax.imshow(self.fundus_image)

        return fig, ax

    def plot(self, layers=None, ax=None, fig=None, disc=True, fovea=True):
        plt.ioff()
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=300)
            ax.set_axis_off()

        if self.image is not None:
            cfi = self.image
            ax.imshow(cfi)
        colors = [(0, 0, 0, 0), (1, 1, 1, 1)]
        cmap = LinearSegmentedColormap.from_list("binary", colors, N=2)

        if layers is not None:
            layers = {l: self.layers[l] for l in layers}
        else:
            layers = self.layers

        for k, l in layers.items():
            l.plot(ax, fig)

        if self.fovea_location is not None and fovea:
            ax.scatter(
                x=self.fovea_location.tuple[1],
                y=self.fovea_location.tuple[0],
                c="green",
                s=2,
            )

        if self.disc is not None and disc:
            ax.imshow(self.disc.mask, cmap=cmap)

        return fig, ax

    @classmethod
    def from_file(
        cls,
        disc_path: Union[str, Path] = None,
        fundus_path: Union[str, Path] = None,
        fovea_location: Tuple[float, float] = None,
        bounds: Bounds = None,
        scaling_factor=1,
        id: Any = None,
        **kwargs,
    ):
        retina = cls(
            disc_path_or_mask=disc_path,
            fundus_path_or_mask=fundus_path,
            fovea_location=fovea_location,
            scaling_factor=scaling_factor,
            bounds=bounds,
            id=id,
        )

        return retina
