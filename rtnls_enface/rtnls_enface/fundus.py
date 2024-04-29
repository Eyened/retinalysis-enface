from pathlib import Path
from typing import Any, Dict, Tuple, Union

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from rtnls_fundusprep.mask_extraction import Bounds

from rtnls_enface.base import EnfaceImage, Layer, Point
from rtnls_enface.disc import OpticDisc
from rtnls_enface.utils.data_loading import open_mask
from rtnls_enface.utils.image import match_resolution


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
        self.id = id
        self.resolution = resolution
        self.load_disc(disc_path_or_mask)
        self.load_fundus_image(fundus_path_or_mask)

        self.layers = layers if layers is not None else dict()
        for layer in self.layers.values():
            layer.retina = self

        self.fovea_location = (
            Point(fovea_location[1], fovea_location[0])
            if fovea_location is not None
            else None
        )
        if bounds is None:
            r = resolution[0] / 2
            bounds = Bounds(*resolution, r, r, r, 0, 0, *resolution)
        self.bounds = bounds
        self.scaling_factor = scaling_factor

    def zone(self, zone_name):
        layers = {
            key: Layer(val.extract_zone(zone_name), self)
            for key, val in self.layers.items()
        }
        return Fundus(layers, self.disc_mask, self.fundus_image)

    def load_fundus_image(self, path_or_array: Union[None, str, np.array]):
        if path_or_array is None:
            self.fundus_image = None
        else:
            if isinstance(path_or_array, np.ndarray):
                self.fundus_image = path_or_array
                self.fundus_path = None
            else:
                im = open_mask(path_or_array)
                self.fundus_image = match_resolution(
                    im, self.resolution, interpolation=cv2.INTER_CUBIC
                )
                self.fundus_path = path_or_array

    def load_disc(self, path_or_array: Union[None, str, np.array]):
        if path_or_array is None:
            self.disc_path = None
            self.disc = None
        else:
            if isinstance(path_or_array, np.ndarray):
                disc_mask = path_or_array
                self.disc_path = None
            else:
                im = open_mask(path_or_array)
                if len(im.shape) == 2:
                    disc_mask = (im > 0).astype(np.uint8).squeeze()
                else:
                    disc_mask = im[:, :, 0].squeeze()
                self.disc_path = path_or_array

            disc_mask = match_resolution(
                disc_mask, self.resolution, interpolation=cv2.INTER_NEAREST
            )
            self.disc = OpticDisc(disc_mask, fundus=self)

    def set_bounds(self, bounds: Bounds):
        self.bounds = bounds

    def variance_of_laplacian(self):
        gray = cv2.cvtColor(self.fundus_image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        mask = self.fundus_mask()

        # Apply the mask to the image
        pixels_within_circle = laplacian[mask == 255]
        return np.var(pixels_within_circle)

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

    def plot(self, layers=None, ax=None, fig=None, disc=True):
        plt.ioff()
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=300)
            ax.set_axis_off()

        if self.fundus_image is not None:
            cfi = self.fundus_image
            ax.imshow(cfi)
        colors = [(0, 0, 0, 0), (1, 1, 1, 1)]
        cmap = LinearSegmentedColormap.from_list("binary", colors, N=2)

        # if disc and self.disc is not None:
        #     cir = plt.Circle(self.disc.center, self.disc.radius,
        #                      color='g', fill=False)
        #     ax.add_patch(cir)
        # ax.plot(self.disc.contour[:,0], self.disc.contour[:,1], marker='o', markersize=1)
        # ax.scatter([self.disc.center[0]], [self.disc.center[1]])

        if layers is not None:
            layers = {l: self.layers[l] for l in layers}
        else:
            layers = self.layers

        for k, l in layers.items():
            l.plot(ax, fig)

        if self.fovea_location is not None:
            ax.scatter(
                x=self.fovea_location.tuple[1],
                y=self.fovea_location.tuple[0],
                c="green",
                s=2,
            )

        etdrs_point = self.etdrs_point()
        if etdrs_point is not None:
            ax.scatter(
                *etdrs_point.tuple_xy,
                c="pink",
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
