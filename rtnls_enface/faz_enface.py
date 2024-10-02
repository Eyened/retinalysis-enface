from pathlib import Path
from typing import Dict

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from rtnls_enface.base import EnfaceImage, Layer
from rtnls_enface.faz import FAZ
from rtnls_enface.utils.data_loading import open_image, open_mask


class FAZEnface(EnfaceImage):
    def __init__(
        self,
        faz_mask: np.array = None,
        layers: Dict[str, Layer] = None,
        resolution=(320, 320),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.faz = FAZ(faz_mask, faz_enface=self) if faz_mask is not None else None
        self.layers = layers if layers is not None else dict()
        for layer in self.layers.values():
            layer.retina = self
        self.resolution = resolution

    def plot(self, layers=None, ax=None, fig=None, disc=True, fovea=True):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=300)
            ax.set_axis_off()

        if self.image is not None:
            cfi = self.image
            ax.imshow(cfi)
        colors = [(0, 0, 0, 0), (1, 1, 1, 1)]
        cmap = LinearSegmentedColormap.from_list("binary", colors, N=2)

        if layers is not None and self.layers is not None:
            layers = {l: self.layers[l] for l in layers}
        else:
            layers = self.layers

        for k, l in layers.items():
            l.plot(ax, fig)

        if self.faz is not None and disc:
            ax.imshow(self.faz.mask, cmap=cmap)

        return fig, ax

    @classmethod
    def from_file(cls, faz_path: str | Path, image_path: str | Path, **kwargs):
        im = open_mask(faz_path)
        if len(im.shape) == 2:
            faz_mask = (im > 0).astype(np.uint8).squeeze()
        else:
            faz_mask = im[:, :, 0].squeeze()

        image = open_image(image_path)
        return cls(faz_mask=faz_mask, image=image, **kwargs)
