from pathlib import Path
from typing import Dict, List, Type, TypeVar

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from rtnls_enface.base import EnfaceImage, Layer, Point
from rtnls_enface.faz import FAZ
from rtnls_enface.grids.base import Grid
from rtnls_enface.grids.etdrs import ETDRSGrid
from rtnls_enface.utils.data_loading import open_image, open_mask

GridClass = TypeVar("T", bound=Grid)

class FAZEnface(EnfaceImage):
    def __init__(
        self,
        faz_mask: np.array = None,
        layers: Dict[str, Layer] = None,
        grids: List[Type[GridClass]] = [ETDRSGrid],
        mm=6, # side length in mm, eg. 3 for 3x3 OCT-A
        laterality=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert self.resolution[0] == self.resolution[1], 'Image not square. FAZEnface currently assumes square images'
        self.faz = FAZ(faz_mask, faz_enface=self) if faz_mask is not None else None
        self.layers = layers if layers is not None else dict()
        for layer in self.layers.values():
            layer.retina = self
        self.laterality = laterality

         # self.grids holds a mapping GridClass -> GridObject
        if self.fovea_location is not None:
            self.grids = {
                cls: cls(
                    self, 
                    resolution=mm / self.resolution[0],
                    mask=None
                ) for cls in grids
            }
        else:
            self.grids = {}

    @property
    def fovea_location(self) -> Point:
        return self.faz.center_of_mass if self. faz is not None else None

    def plot(self, layers=None, ax=None, fig=None, faz=True):
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=300)
            ax.set_axis_off()

        if self.image is not None:
            ax.imshow(self.image)
        colors = [(0, 0, 0, 0), (1, 1, 1, 1)]
        cmap = LinearSegmentedColormap.from_list("binary", colors, N=2)

        if layers is not None and self.layers is not None:
            layers = {l: self.layers[l] for l in layers}
        else:
            layers = self.layers

        for k, l in layers.items():
            l.plot(ax, fig)

        if self.faz is not None and faz:
            ax.imshow(self.faz.mask, cmap=cmap)

        return ax

    @classmethod
    def from_file(cls, faz_path: str | Path, image_path: str | Path, **kwargs):
        im = open_mask(faz_path)
        if len(im.shape) == 2:
            faz_mask = (im > 0).astype(np.uint8).squeeze()
        else:
            faz_mask = im[:, :, 0].squeeze()

        image = open_image(image_path)
        return cls(faz_mask=faz_mask, image=image, **kwargs)
