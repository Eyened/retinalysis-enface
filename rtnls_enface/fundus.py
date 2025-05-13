from __future__ import annotations

import warnings
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Type, TypeVar, Union

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from rtnls_enface.base import EnfaceImage, Laterality, Layer, Point
from rtnls_enface.disc import InvalidDiscMaskException, OpticDisc
from rtnls_enface.grids.base import Grid
from rtnls_enface.grids.etdrs import ETDRSGrid
from rtnls_enface.utils.data_loading import open_binary_mask, open_mask
from rtnls_enface.utils.image import match_resolution

if TYPE_CHECKING:
    from rtnls_fundusprep.mask_extraction import CFIBounds as Bounds


GridClass = TypeVar("T", bound=Grid)


class Fundus(EnfaceImage):
    def __init__(
        self,
        disc_path_or_mask: Union[str, Path, np.array] = None,
        fundus_path_or_mask: Union[str, Path, np.array] = None,
        layers: Dict[str, Layer] = None,
        fovea_location: Tuple[float, float] = None,
        bounds: Bounds = None,
        grids: List[Type[GridClass]] = [ETDRSGrid],  # list of grid classes
        scaling_factor=1.0,
        resolution=(1024, 1024),
        id: Any = None,
    ):
        image = self._load_fundus_image(fundus_path_or_mask)
        super().__init__(image, id)

        self.name = "image"
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
        # self.grids holds a mapping GridClass -> GridObject
        if self.disc is not None and self.fovea_location is not None:
            self.grids = {
                cls: cls(
                    self,
                    resolution=4.75 / self.disc_fovea_distance,
                    mask=self.bounds.make_binary_mask()
                    if self.bounds is not None
                    else None,
                )
                for cls in grids
            }
        else:
            self.grids = {}

    @cached_property
    def grayscale(self) -> np.ndarray:
        return np.dot(self.image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

    @cached_property
    def contrast_enhanced(self) -> np.ndarray:
        if self.bounds is None:
            raise RuntimeError("Bounds must be set for contrast enhancing")
        im = self.bounds.make_contrast_enhanced_res256(self.image)
        return np.dot(im, [0.299, 0.587, 0.114]).astype(np.uint8)

    def get_layers(self, layer_type):
        return [
            layer for layer in self.layers.values() if isinstance(layer, layer_type)
        ]

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
            try:
                self.disc = OpticDisc(disc_mask, fundus=self)
            except InvalidDiscMaskException:
                warnings.warn(f"Invalid disc mask. Setting disc to None")
                self.disc = None

    @cached_property
    def disc_fovea_distance(self):
        if self.disc is None or self.fovea_location is None:
            raise ValueError("Disc or fovea location not set")
        return self.disc.center_of_mass.distance_to(self.fovea_location)

    @cached_property
    def laterality(self):
        if self.disc is None or self.fovea_location is None:
            return None
        if self.disc.center_of_mass.x < self.fovea_location.x:
            return Laterality.L
        return Laterality.R

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

    def plot(self, layers=None, disc=True, fovea=True, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)
            ax.imshow(np.zeros(self.resolution))
            ax.set_axis_off()

        if self.image is not None:
            ax.imshow(self.image)

        colors = [(0, 0, 0, 0), (1, 1, 1, 1)]
        cmap = LinearSegmentedColormap.from_list("binary", colors, N=2)

        if isinstance(layers, list):
            layers = {l: self.layers[l] for l in layers}
        elif layers is False:
            layers = {}
        else:
            layers = self.layers

        for k, l in layers.items():
            l.plot(ax=ax, **kwargs)

        if self.fovea_location is not None and fovea:
            ax.scatter(
                x=self.fovea_location.tuple[1],
                y=self.fovea_location.tuple[0],
                c="white",
                marker="x",
                s=8,
            )

        if self.disc is not None and disc:
            ax.imshow(self.disc.mask, cmap=cmap)

        return ax

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
