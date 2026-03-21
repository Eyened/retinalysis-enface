from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple, Type, TypeVar, cast

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from rtnls_enface.base import EnfaceImage, Layer, Point
from rtnls_enface.faz import FAZ
from rtnls_enface.grids.base import Grid, GridField
from rtnls_enface.grids.etdrs import ETDRSGrid
from rtnls_enface.grids.specifications import (
    BaseGridFieldSpecification,
    BaseGridSpecification,
    ETDRSGridSpecification,
)
from rtnls_enface.utils.data_loading import open_image, open_mask

GridClass = TypeVar("T", bound=Grid)

_FAZ_DEFAULT_GRID_CLASSES: Tuple[Type[Grid], ...] = (ETDRSGrid,)
_FAZ_DEFAULT_SPEC_BY_GRID: Dict[Type[Grid], BaseGridSpecification] = {
    ETDRSGrid: ETDRSGridSpecification(),
}


class FAZEnface(EnfaceImage):
    def __init__(
        self,
        faz_mask: np.array = None,
        layers: Dict[str, Layer] = None,
        grids: Sequence[Type[GridClass]] | None = None,
        mm=6,  # side length in mm, eg. 3 for 3x3 OCT-A
        laterality=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert self.resolution[0] == self.resolution[1], (
            "Image not square. FAZEnface currently assumes square images"
        )
        self.faz = FAZ(faz_mask, faz_enface=self) if faz_mask is not None else None
        self.layers = layers if layers is not None else dict()
        for layer in self.layers.values():
            layer.retina = self
        self.laterality = laterality
        self._scan_mm = mm

        self._grid_cache: Dict[BaseGridSpecification, Grid] = {}
        self._legacy_grid_cache: Dict[Type[Grid], Grid] = {}
        grid_classes = tuple(grids) if grids is not None else _FAZ_DEFAULT_GRID_CLASSES
        if self.fovea_location is not None:
            self._grid_classes: Tuple[Type[Grid], ...] = grid_classes
        else:
            self._grid_classes = tuple()
        self.grids: Mapping[Type[Grid], Grid] = _LazyFAZGridMapping(self)

    @property
    def fovea_location(self) -> Point:
        return self.faz.center_of_mass if self.faz is not None else None

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

    def _grid_resolution(self) -> float:
        return float(self._scan_mm) / float(self.resolution[0])

    def _grid_mask(self):
        return None

    def get_grid(self, spec: BaseGridSpecification[GridClass]) -> GridClass:
        grid = self._grid_cache.get(spec)
        if grid is None:
            grid = spec.grid_cls(
                self,
                resolution=self._grid_resolution(),
                mask=self._grid_mask(),
                **spec.init_kwargs(),
            )
            self._grid_cache[spec] = grid
        return cast(GridClass, grid)

    def get_grid_field(self, field_spec: BaseGridFieldSpecification) -> GridField:
        grid = self.get_grid(field_spec.grid_spec)
        return grid.field(field_spec.field)

    def _legacy_spec_for_class(
        self, grid_cls: Type[Grid]
    ) -> BaseGridSpecification | None:
        return _FAZ_DEFAULT_SPEC_BY_GRID.get(grid_cls)

    def _get_legacy_grid(self, grid_cls: Type[GridClass]) -> GridClass:
        grid = self._legacy_grid_cache.get(grid_cls)
        if grid is not None:
            return cast(GridClass, grid)

        spec = self._legacy_spec_for_class(grid_cls)
        if spec is not None:
            grid = self.get_grid(spec)
        else:
            grid = grid_cls(
                self,
                resolution=self._grid_resolution(),
                mask=self._grid_mask(),
            )
        self._legacy_grid_cache[grid_cls] = grid
        return cast(GridClass, grid)


class _LazyFAZGridMapping(Mapping[Type[Grid], Grid]):
    """Lazy mapping replicating the legacy `FAZEnface.grids` API."""

    def __init__(self, retina: FAZEnface) -> None:
        self._retina = retina

    def __getitem__(self, grid_cls: Type[Grid]) -> Grid:
        return self._retina._get_legacy_grid(grid_cls)

    def __iter__(self):
        return iter(self._retina._grid_classes)

    def __len__(self) -> int:
        return len(self._retina._grid_classes)
